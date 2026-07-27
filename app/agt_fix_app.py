"""AGT Polygon Fixer - guided GUI over the WV3 alignment/QA/topology pipeline.

Run:  streamlit run agt_fix_app.py
A tabbed wizard: Data -> Polygons -> Configure & Run -> Review -> Topology -> Export.
Heavy work runs in-process (this env has rasterio/onnxruntime/geopandas); the
Review tab launches FiftyOne in its own 3.11 venv as a subprocess and embeds it.
"""
import glob
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import streamlit as st
from rasterio.enums import Resampling
from rasterio.transform import array_bounds

import app_pipeline as ap

# Streamlit Cloud's on-push redeploy does not always restart the underlying
# Python process -- when it only reruns the script, an already-imported module
# keeps running its OLD in-memory code even after `git pull` updates the file
# on disk (this bit us: app_pipeline.py picked up a fix immediately, but
# buildseg_pipeline_tile.py -- imported deeper in the chain -- kept running
# the stale version until a manual "Reboot app"). Force a reload of this app's
# own modules on every script run, leaves first so each module's own `import X
# as Y` re-binds to the freshly reloaded version, so a push always takes
# effect without relying on a manual reboot.
for _mod_name in ("regularize_lib", "topology_tools", "confidence_score",
                  "buildseg_pipeline_tile", "fix_agt_polygons", "app_pipeline"):
    _mod = sys.modules.get(_mod_name)
    if _mod is not None:
        try:
            importlib.reload(_mod)
        except Exception:
            pass

# Mirror Streamlit Cloud secrets (App settings -> Secrets) into real environment
# variables, so the framework-agnostic buildseg_pipeline_tile.find_model() can
# read them uniformly whether it's running here, from the CLI, or locally.
for _key in ("AGT_ONNX", "AGT_ONNX_URL", "AGT_ONNX_CACHE"):
    if not os.environ.get(_key):
        try:
            _val = st.secrets.get(_key)
        except Exception:
            _val = None
        if _val:
            os.environ[_key] = _val

FIFTYONE_PY = os.environ.get("AGTFIX_FIFTYONE_PY",
                             r"C:\Users\phila\venvs\fiftyone311\Scripts\python.exe")
FIFTYONE_PORT = 5151
DEFAULT_OUT = os.path.join(os.path.dirname(ap.NB), "Data", "Outputs", "agt_fix", "app_runs")
CLASS_COLORS = {"keep": "#39d353", "review": "#ff9f0a", "drop": "#ff453a",
                "duplicate": "#bf5af2", "edge": "#0a84ff"}
RASTER_EXTS = ("*.tif", "*.tiff", "*.jp2", "*.img", "*.vrt")


def find_rasters(folder):
    files = []
    for ext in RASTER_EXTS:
        files += glob.glob(os.path.join(folder, ext))
    return sorted(files)


def clean_path(p):
    """Strip whitespace and surrounding quotes (Windows 'Copy as path' wraps
    paths in double quotes, which break os/rasterio calls)."""
    return p.strip().strip('"').strip("'").strip() if p else ""

st.set_page_config(page_title="AGT Polygon Fixer", page_icon="🛰️", layout="wide")
S = st.session_state
S.setdefault("raster", "")
S.setdefault("polygons", "")
S.setdefault("result", None)
S.setdefault("gpkg", None)
S.setdefault("fo_proc", None)


def upload_dir():
    """One scratch folder per browser session, reused across reruns."""
    S.setdefault("_upload_dir", None)
    if not S["_upload_dir"]:
        S["_upload_dir"] = tempfile.mkdtemp(prefix="agtfix_")
    return S["_upload_dir"]


def save_upload(file, subdir=""):
    """Write an uploaded file to disk (rasterio/geopandas both need a real
    path, not an in-memory buffer -- shapefiles especially, since the .shp
    driver opens its .shx/.dbf siblings by path)."""
    d = os.path.join(upload_dir(), subdir)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, file.name)
    with open(path, "wb") as f:
        f.write(file.getbuffer())
    return path


def resolve_uploaded_polygons(ups):
    """Persist uploaded polygon file(s) and return one path geopandas can
    open: a .gpkg/.geojson directly, the contents of a zipped Shapefile, or a
    bare .shp whose .shx/.dbf sidecars were uploaded alongside it.

    The scratch folder is wiped at the start of every call: a Shapefile set is
    only ever known-consistent as of the current upload, and re-scanning a
    folder that also held an earlier, unrelated upload from the same browser
    session risks silently resolving to a stale leftover file instead.
    """
    d = os.path.join(upload_dir(), "polygons")
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

    saved = []
    for u in ups:
        if u.name.lower().endswith(".zip"):
            zpath = os.path.join(d, u.name)
            with open(zpath, "wb") as f:
                f.write(u.getbuffer())
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(d)
                saved += [os.path.join(d, n) for n in zf.namelist() if not n.endswith("/")]
            os.remove(zpath)
        else:
            path = os.path.join(d, u.name)
            with open(path, "wb") as f:
                f.write(u.getbuffer())
            saved.append(path)

    gpkg_or_json = next((p for p in saved
                        if p.lower().endswith((".gpkg", ".geojson", ".json"))), None)
    if gpkg_or_json:
        return gpkg_or_json
    shp = next((p for p in saved if p.lower().endswith(".shp")), None)
    if shp:
        base = os.path.splitext(shp)[0]
        missing = [ext for ext in (".shx", ".dbf") if not os.path.exists(base + ext)]
        if missing:
            st.warning("A Shapefile needs its sidecar files too -- missing "
                      f"{', '.join(missing)}. Select the .shp together with its "
                      ".shx/.dbf/.prj, or upload a single .zip of the whole set.")
            return ""
        return shp
    if saved:
        st.warning("Upload a .gpkg/.geojson, a zipped Shapefile (.zip), or the "
                  "full Shapefile file set (.shp + .shx + .dbf).")
    return ""


def thumb(raster_path, max_px=1000):
    with rasterio.open(raster_path) as r:
        scale = max(1, int(max(r.width, r.height) / max_px))
        out = (3, r.height // scale, r.width // scale)
        arr = r.read([1, 2, 3], out_shape=out, resampling=Resampling.bilinear)
    return np.ascontiguousarray(arr.transpose(1, 2, 0))


def overlay_fig(result, max_px=1400):
    img, tr = result["img"], result["transform"]
    H, W = img.shape[:2]
    scale = max(1, int(max(H, W) / max_px))
    disp = img[::scale, ::scale]
    l, b, rt, t = array_bounds(H, W, tr)
    fig, ax = plt.subplots(figsize=(13, 8), dpi=110)
    ax.imshow(disp, extent=(l, rt, b, t))
    qa = result["qa"]
    for cls, col in CLASS_COLORS.items():
        sub = qa[qa.qa_class == cls]
        if len(sub):
            sub.boundary.plot(ax=ax, color=col, linewidth=0.6,
                              label=f"{cls} ({len(sub)})")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


st.title("🛰️ AGT Polygon Fixer")
st.caption("Align, QA and clean building polygons against WorldView-3 imagery.")

tabs = st.tabs(["1 · WV3 data", "2 · Polygons", "3 · Configure & run",
                "4 · Review", "5 · Topology", "6 · Export"])

# ---------------------------------------------------------------- Tab 1: raster
with tabs[0]:
    st.subheader("Step 1 — WorldView-3 imagery")
    mode = st.radio("Where is the imagery?",
                    ["Type a path on this machine", "Upload from your device"],
                    key="raster_source_mode", horizontal=True)
    S["raster"] = ""

    if mode == "Type a path on this machine":
        st.write("Point to the WV3 raster **file** (GeoTIFF / JP2), or a folder to pick "
                 "from. Rasters are large, so the app reads from disk rather than uploading.")
        st.text_input("Raster path (file or folder)", key="raster_path_input",
                      placeholder=r"...\Clipped_Training\Luanda_Informal_1.tif")
        typed = clean_path(S.get("raster_path_input", ""))
        if typed and os.path.isdir(typed):
            rasters = find_rasters(typed)
            if rasters:
                st.info(f"That's a folder — pick a raster file ({len(rasters)} found):")
                choice = st.selectbox("Raster file", rasters,
                                      format_func=os.path.basename)
                S["raster"] = choice
            else:
                st.warning("No raster files (.tif/.tiff/.jp2/.img/.vrt) found in that folder.")
        elif typed and os.path.isfile(typed):
            S["raster"] = typed
        elif typed:
            st.warning("Path not found.")
    else:
        st.write("Upload a WV3 raster (GeoTIFF / JP2). Uploads are capped by this "
                 "deployment's upload-size limit and held in a temporary folder for "
                 "this session only — for very large rasters, run the app locally and "
                 "use a path instead.")
        up = st.file_uploader("Raster file", type=["tif", "tiff", "jp2", "img"],
                              key="raster_upload")
        if up is not None:
            S["raster"] = save_upload(up, "raster")

    if S["raster"]:
        try:
            with rasterio.open(S["raster"]) as r:
                mp = r.width * r.height / 1e6
                gb = ap.est_peak_gb(r.height, r.width)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Width", f"{r.width:,}")
                c2.metric("Height", f"{r.height:,}")
                c3.metric("Megapixels", f"{mp:,.0f}")
                c4.metric("Pixel size", f"{abs(r.transform.a):.3f} m")
            S["raster_mp"] = mp
            st.image(thumb(S["raster"]), caption=os.path.basename(S["raster"]),
                     use_container_width=True)
            if mp > ap.DEFAULT_MAX_MP:
                st.warning(f"This raster is large ({mp:,.0f} MP, ~{gb:.1f} GB peak "
                           f"RAM). The pipeline holds the whole raster in memory — "
                           "clip to an area of interest, or use the override in "
                           "Step 3 only if this machine has enough RAM.")
            else:
                st.success("Raster loaded. Continue to Step 2.")
        except Exception as e:
            st.error(f"Could not read this as a raster file: {e}\n\n"
                     "Make sure the path is a single .tif/.jp2 file, not a folder.")

# -------------------------------------------------------------- Tab 2: polygons
with tabs[1]:
    st.subheader("Step 2 — Polygons to fix")
    mode2 = st.radio("Where are the polygons?",
                     ["Type a path on this machine", "Upload from your device"],
                     key="poly_source_mode", horizontal=True)
    S["polygons"] = ""

    if mode2 == "Type a path on this machine":
        st.write("Point to the polygon layer (GeoPackage / Shapefile). Only features "
                 "inside the raster footprint are loaded.")
        S["polygons"] = clean_path(st.text_input(
            "Polygon path", key="poly_path_input",
            placeholder=r"...\Luanda_Polygons\agt_luanda_full.gpkg"))
    else:
        st.write("Upload a **GeoPackage** (.gpkg), a **zipped Shapefile** (.zip), or the "
                 "individual Shapefile parts (.shp + .shx + .dbf + .prj) selected together.")
        ups = st.file_uploader(
            "Polygon file(s)", accept_multiple_files=True,
            type=["gpkg", "geojson", "json", "zip", "shp", "shx", "dbf", "prj",
                 "cpg", "sbn", "sbx", "qpj"],
            key="poly_upload")
        if ups:
            S["polygons"] = resolve_uploaded_polygons(ups)

    if S["polygons"] and os.path.exists(S["polygons"]) and S["raster"] and os.path.exists(S["raster"]):
        if st.button("Count polygons in this raster window"):
            try:
                with rasterio.open(S["raster"]) as r:
                    gdf = ap.load_polygons(S["polygons"], r.bounds, r.crs)
                S["poly_preview_n"] = len(gdf)
            except Exception as e:
                st.error(f"Could not read polygons: {e}")
        if S.get("poly_preview_n") is not None:
            st.success(f"{S['poly_preview_n']:,} polygons fall within the raster. Continue to Step 3.")
    elif S["polygons"]:
        st.warning("Set a valid raster (Step 1) and polygon path.")

# ------------------------------------------------------- Tab 3: configure & run
with tabs[2]:
    st.subheader("Step 3 — Choose corrections and run")
    st.write("**Corrections to apply** — tick the stages to run.")
    c1, c2 = st.columns(2)
    dedup = c1.checkbox("Remove duplicates", value=True,
                        help="Drop polygons swallowed by a larger one or near-identical twins.")
    align = c1.checkbox("Correct geometric shift (alignment)", value=True,
                        help="Iterative grid alignment against the imagery.")
    refine = c2.checkbox("Per-polygon refinement", value=True,
                         help="Nudge individual polygons onto their roof (needs alignment on).")
    planarize = c2.checkbox("Fix overlaps (planarize)", value=True,
                            help="Resolve shared/double-counted area in the retained set.")
    st.divider()
    st.write("**Classification thresholds** — how keep / review / drop is decided.")
    d = ap.default_options()
    c1, c2, c3 = st.columns(3)
    keep_mean = c1.slider("Keep if mean prob ≥", 0.0, 1.0, float(d["keep_mean"]), 0.05)
    drop_mean = c2.slider("Drop if mean prob <", 0.0, 1.0, float(d["drop_mean"]), 0.05)
    edge_frac = c3.slider("Edge if < frac inside", 0.0, 1.0, float(d["edge_frac"]), 0.05)
    keep_cover = c1.slider("Keep if coverage ≥", 0.0, 1.0, float(d["keep_cover"]), 0.05)
    drop_cover = c2.slider("Drop if coverage <", 0.0, 1.0, float(d["drop_cover"]), 0.05)

    ready = all(os.path.exists(p) for p in (S["raster"] or "x", S["polygons"] or "x")) \
        and S["raster"] and S["polygons"]
    if not ready:
        st.info("Complete Steps 1 and 2 first.")
    allow_large = False
    if ready and S.get("raster_mp", 0) > ap.DEFAULT_MAX_MP:
        allow_large = st.checkbox(
            f"⚠ Process anyway — raster is {S['raster_mp']:,.0f} MP "
            f"(limit {ap.DEFAULT_MAX_MP} MP). Only tick this if this machine "
            "has plenty of RAM.")
    if st.button("▶ Run pipeline", type="primary", disabled=not ready):
        opts = dict(dedup=dedup, align=align, refine=refine, planarize=planarize,
                    keep_mean=keep_mean, keep_cover=keep_cover,
                    drop_mean=drop_mean, drop_cover=drop_cover, edge_frac=edge_frac)
        bar = st.progress(0.0, text="Starting…")

        def cb(msg, frac=None):
            bar.progress(min(frac or 0.0, 1.0), text=msg)
        try:
            with st.spinner("Running…"):
                S["result"] = ap.run_pipeline(S["raster"], S["polygons"],
                                              options=opts, out_dir=DEFAULT_OUT,
                                              progress=cb, allow_large=allow_large)
                S["gpkg"] = None
            bar.progress(1.0, text="Done")
            s = S["result"]["stats"]
            st.success(f"Processed {s['n_in']:,} polygons in {s['seconds']}s "
                       f"({s['n_iters']} alignment iters, {s['n_refined']} refined).")
            st.write("**Class counts:**", s["counts"])
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.exception(e)

# ------------------------------------------------------------- Tab 4: review
with tabs[3]:
    st.subheader("Step 4 — Review results")
    if S["result"] is None:
        st.info("Run the pipeline (Step 3) first.")
    else:
        res = S["result"]
        s = res["stats"]
        cols = st.columns(len(CLASS_COLORS))
        for col, (cls, _) in zip(cols, CLASS_COLORS.items()):
            col.metric(cls, s["counts"].get(cls, 0))
        st.pyplot(overlay_fig(res), use_container_width=True)
        with st.expander("Attribute table (aligned + scored)"):
            show = [c for c in ["qa_class", "mean_prob", "cover", "shift_m",
                                "grid_m", "refine_m", "frac_in"] if c in res["qa"].columns]
            pick = st.multiselect("Filter classes", list(CLASS_COLORS),
                                  default=["drop", "review"])
            df = res["qa"][res["qa"].qa_class.isin(pick)] if pick else res["qa"]
            st.dataframe(df[show].reset_index(drop=True), height=300)

        st.divider()
        st.write("**Visual triage in FiftyOne** — inspect and confirm keep/drop on the chips.")
        cc1, cc2 = st.columns([1, 3])
        if cc1.button("Launch FiftyOne review"):
            launcher = os.path.join(os.path.dirname(__file__), "launch_fiftyone.py")
            if not os.path.exists(FIFTYONE_PY):
                st.error(f"FiftyOne venv not found at {FIFTYONE_PY}")
            else:
                with st.spinner("Preparing image chips for review…"):
                    if not S["gpkg"]:
                        S["gpkg"] = ap.save_outputs(res, DEFAULT_OUT)
                    manifest = ap.prepare_fiftyone_package(
                        S["gpkg"], S["raster"], DEFAULT_OUT)
                S["fo_proc"] = subprocess.Popen(
                    [FIFTYONE_PY, launcher, "--manifest", manifest,
                     "--port", str(FIFTYONE_PORT)])
                st.success("FiftyOne is starting (first launch can take ~30 s)…")
        cc2.markdown(f"[Open FiftyOne in a new tab](http://localhost:{FIFTYONE_PORT})")
        if S["fo_proc"] is not None:
            st.components.v1.iframe(f"http://localhost:{FIFTYONE_PORT}", height=760)

# ------------------------------------------------------------ Tab 5: topology
with tabs[4]:
    st.subheader("Step 5 — Topology (overlaps)")
    if S["result"] is None:
        st.info("Run the pipeline (Step 3) first.")
    else:
        s = S["result"]["stats"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retained (clean)", s["n_clean"])
        c2.metric("Polygons clipped", s["planar_clipped"])
        c3.metric("Overlap removed", f"{s['overlap_removed_m2']:,.0f} m²")
        c4.metric("Residual overlap", f"{s['residual_overlap_m2']:.1f} m²")
        if s["residual_overlap_m2"] <= 1.0:
            st.success("Retained layer is planar — no shared/double-counted area.")
        else:
            st.warning("Residual overlap remains; check planarize settings.")
        st.caption(f"{s['planar_emptied']} polygon(s) fully consumed by a higher-evidence "
                   "neighbour (removed as double-counts).")

# -------------------------------------------------------------- Tab 6: export
with tabs[5]:
    st.subheader("Step 6 — Export")
    if S["result"] is None:
        st.info("Run the pipeline (Step 3) first.")
    else:
        st.write("Writes three layers: **agt_qa** (all polygons + QA attributes), "
                 "**agt_clean** (planar keep+review+edge deliverable), "
                 "**agt_original** (pre-shift geometry).")
        out_dir = clean_path(st.text_input("Output folder", value=DEFAULT_OUT))
        if st.button("💾 Write GeoPackage", type="primary"):
            try:
                os.makedirs(out_dir, exist_ok=True)
                S["gpkg"] = ap.save_outputs(S["result"], out_dir)
                st.success(f"Saved: {S['gpkg']}")
            except Exception as e:
                st.error(f"Save failed: {e}")
        if S["gpkg"] and os.path.exists(S["gpkg"]):
            with open(S["gpkg"], "rb") as fh:
                st.download_button("Download GeoPackage", fh,
                                   file_name=os.path.basename(S["gpkg"]))
