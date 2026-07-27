"""Pipeline entry point for the AGT Polygon Fixer app.

Wraps the tested stage functions in ``fix_agt_polygons`` (dedup -> iterative grid
alignment -> gated per-polygon refinement -> evidence classification -> planarize)
so they run on an ARBITRARY WV3 raster + polygon layer chosen in the GUI, rather
than the fixed tile-name / bbox convention of the CLI. The stage maths is
unchanged; this only handles I/O, stage toggles, adjustable thresholds and
progress callbacks.
"""
import os
import sys
import time

import numpy as np
import geopandas as gpd
import pyogrio
import rasterio
from pyproj import CRS, Transformer
from rasterio.transform import array_bounds
from shapely.affinity import translate

# Locate the pipeline modules. Works in both layouts:
#   dev package:  App/app_pipeline.py  -> ../Notebooks
#   bundled:      app/app_pipeline.py  -> ..  (modules sit next to the app folder)
_HERE = os.path.dirname(os.path.abspath(__file__))
NB = next((c for c in (os.path.join(os.path.dirname(_HERE), "Notebooks"),
                       os.path.dirname(_HERE))
           if os.path.exists(os.path.join(c, "fix_agt_polygons.py"))),
          os.path.dirname(_HERE))
if NB not in sys.path:
    sys.path.insert(0, NB)

import fix_agt_polygons as fx           # noqa: E402
import buildseg_pipeline_tile as bpt    # noqa: E402
import topology_tools as tt             # noqa: E402


def _noop(msg, frac=None):
    pass


# peak RAM is dominated by a handful of full-raster arrays held at once
# (img 3 B + prob 4 B + prob_eff 4 B + rasterised mask 4 B + veg 1 B + overhead)
BYTES_PER_PX = 20
DEFAULT_MAX_MP = 400          # ~8 GB peak; above this the run is blocked unless overridden


def est_peak_gb(height, width):
    return height * width * BYTES_PER_PX / 1e9


def raster_megapixels(raster_path):
    with rasterio.open(raster_path) as r:
        return r.width * r.height / 1e6, (r.height, r.width)


def default_options():
    return dict(
        dedup=True, align=True, refine=False, planarize=True,   # refine OFF (hurts accuracy)
        keep_mean=fx.KEEP_MEAN, keep_cover=fx.KEEP_COVER,
        drop_mean=fx.DROP_MEAN, drop_cover=fx.DROP_COVER,
        edge_frac=fx.EDGE_FRAC, topo_policy="gated",   # gated default (keeps evidence-backed slivers)
    )


def load_polygons(agt_path, raster_bounds, raster_crs, layer=0):
    """Read polygons intersecting the raster footprint and reproject to the
    raster CRS. Uses a bbox filter (in the layer's own CRS) so huge national
    files stream only the relevant window."""
    info = pyogrio.read_info(agt_path, layer=layer)
    src_crs = CRS.from_user_input(info["crs"]) if info["crs"] else CRS.from_user_input(raster_crs)
    rc = CRS.from_user_input(raster_crs)
    l, b, r, t = raster_bounds
    if src_crs == rc:
        bbox = (l, b, r, t)
    else:
        tr = Transformer.from_crs(rc, src_crs, always_xy=True)
        xs, ys = zip(*[tr.transform(x, y) for x in (l, r) for y in (b, t)])
        bbox = (min(xs), min(ys), max(xs), max(ys))
    gdf = pyogrio.read_dataframe(agt_path, layer=layer, bbox=bbox)
    if gdf.crs is None:
        gdf.set_crs(src_crs, inplace=True)
    gdf = gdf.to_crs(rc)
    gdf["geometry"] = gdf.geometry.make_valid()
    gdf = gdf[~gdf.geometry.is_empty & (gdf.geometry.area > 0)].reset_index(drop=True)
    return gdf


def _classify_vec(gdf, o):
    """Vectorised keep/review/drop/edge/duplicate using GUI thresholds."""
    cls = np.where(gdf.qa_class.values == "duplicate", "duplicate",
          np.where(gdf.frac_in.values < o["edge_frac"], "edge",
          np.where((gdf.mean_prob.values >= o["keep_mean"]) |
                   (gdf.cover.values >= o["keep_cover"]), "keep",
          np.where((gdf.mean_prob.values < o["drop_mean"]) &
                   (gdf.cover.values < o["drop_cover"]), "drop", "review"))))
    return cls


def run_pipeline(raster_path, agt_path, options=None, out_dir=None,
                 prob_cache=True, progress=None, max_megapixels=DEFAULT_MAX_MP,
                 allow_large=False):
    """Run the fix pipeline on one raster + polygon layer.

    Returns a dict: qa (full GeoDataFrame, aligned+scored+classified, unplanarised),
    clean (planarised keep+review+edge), original (pre-shift geometry), img (HWC
    uint8), prob (float32), transform, crs, bounds, stats (per-stage numbers).

    Raises MemoryError early (before any heavy work) if the raster is larger than
    ``max_megapixels`` and ``allow_large`` is False, so a whole WV3 tile fails with
    guidance instead of dying mid-inference. The pipeline holds several full-raster
    arrays in RAM, so it targets clipped AOIs, not full Maxar tiles.
    """
    o = default_options()
    if options:
        o.update(options)
    progress = progress or _noop
    out_dir = out_dir or fx.OUTDIR
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(raster_path))[0]
    t0 = time.time()

    progress("Reading raster", 0.02)
    with rasterio.open(raster_path) as r:
        H, W = r.height, r.width
        mp = W * H / 1e6
        if mp > max_megapixels and not allow_large:
            raise MemoryError(
                f"Raster is {W:,}x{H:,} = {mp:,.0f} megapixels (est. peak "
                f"~{est_peak_gb(H, W):.1f} GB RAM), above the {max_megapixels} MP "
                "limit. Use a clipped AOI, or tick 'Process anyway' if this "
                "machine has enough memory. Full Maxar tiles are not supported "
                "in-memory — clip to an area of interest first.")
        img = r.read([1, 2, 3]).transpose(1, 2, 0)
        transform, crs, bounds = r.transform, r.crs, r.bounds
    px_area = abs(transform.a * transform.e)

    # --- building probability (cached GeoTIFF next to outputs) ---
    prob_path = os.path.join(out_dir, f"prob_{stem}.tif")
    if prob_cache and os.path.exists(prob_path):
        progress("Loading cached probability raster", 0.10)
        with rasterio.open(prob_path) as pr:
            prob = pr.read(1)
    else:
        progress("Running building detection (buildseg SegFormer)", 0.10)
        import onnxruntime as ort
        sess = ort.InferenceSession(bpt.find_model(), providers=["CPUExecutionProvider"])
        prob = bpt.bseg_prob(img, sess, sess.get_inputs()[0].name)
        with rasterio.open(prob_path, "w", driver="GTiff", width=W, height=H,
                           count=1, dtype="float32", transform=transform, crs=crs,
                           compress="lzw") as w:
            w.write(prob.astype("float32"), 1)
    f = img.astype("float32")
    veg = (2 * f[:, :, 1] - f[:, :, 0] - f[:, :, 2]) > fx.EXG_VEG
    prob_eff = np.where(veg, 0.0, prob).astype("float32")

    progress("Loading polygons", 0.30)
    gdf = load_polygons(agt_path, bounds, crs)
    n_in = len(gdf)
    gdf["qa_class"] = ""
    orig_geoms = list(gdf.geometry)
    inv = ~transform
    cents0 = np.array([inv * (g.centroid.x, g.centroid.y) for g in orig_geoms])

    # --- 1. duplicates ---
    if o["dedup"]:
        progress("Flagging duplicates", 0.38)
        dup = fx.flag_duplicates(gdf)
    else:
        dup = np.zeros(n_in, bool)
    gdf.loc[dup, "qa_class"] = "duplicate"

    # --- 2a. iterative grid alignment ---
    cum = np.zeros((n_in, 2))
    n_iters = 0
    if o["align"]:
        for it in range(fx.MAX_ITERS):
            progress(f"Aligning (grid iteration {it + 1})", 0.42 + 0.04 * it)
            shifted = [translate(g, xoff=cum[i, 0] * transform.a,
                                 yoff=cum[i, 1] * transform.e)
                       for i, g in enumerate(orig_geoms)]
            live = [shifted[i] for i in range(n_in) if not dup[i]]
            cells, _ = fx.displacement_grid(live, prob_eff, transform, (H, W))
            if cells is None or len(cells) == 0:
                break
            n_iters = it + 1
            res = np.median(cells[:, 2:4], axis=0)
            cum += fx.per_poly_offsets(cells, cents0 + cum)
            if np.hypot(*res) < fx.CONV_PX:
                break
    grid_geoms = [translate(g, xoff=cum[i, 0] * transform.a,
                            yoff=cum[i, 1] * transform.e)
                  for i, g in enumerate(orig_geoms)]

    # --- 2b. gated per-polygon refinement ---
    refine = np.zeros((n_in, 2))
    applied = np.zeros(n_in, bool)
    if o["align"] and o["refine"]:
        progress("Refining individual polygons", 0.62)
        om = np.array([fx.zonal(g, prob_eff, transform, px_area) for g in orig_geoms])
        gm = np.array([fx.zonal(g, prob_eff, transform, px_area) for g in grid_geoms])
        building = ((np.nan_to_num(om[:, 0]) >= fx.REFINE_EVID_MEAN) |
                    (np.nan_to_num(om[:, 1]) >= fx.REFINE_EVID_COVER) |
                    (np.nan_to_num(gm[:, 0]) >= fx.REFINE_EVID_MEAN) |
                    (np.nan_to_num(gm[:, 1]) >= fx.REFINE_EVID_COVER))
        eligible = building & ~dup
        refine, _, applied = fx.refine_offsets(grid_geoms, prob_eff, transform, eligible)

    total = cum + refine
    gdf["dx_m"] = total[:, 0] * transform.a
    gdf["dy_m"] = total[:, 1] * transform.e
    gdf["grid_m"] = np.hypot(cum[:, 0] * transform.a, cum[:, 1] * transform.e).round(3)
    gdf["refine_m"] = np.hypot(refine[:, 0] * transform.a, refine[:, 1] * transform.e).round(3)
    gdf["refined"] = applied
    aligned = [translate(g, xoff=dx, yoff=dy)
               for g, dx, dy in zip(orig_geoms, gdf.dx_m, gdf.dy_m)]

    # --- 3. evidence scoring + classification ---
    progress("Scoring polygons against imagery", 0.78)
    stats_zonal = np.array([fx.zonal(g, prob_eff, transform, px_area) for g in aligned])
    gdf["mean_prob"], gdf["cover"], gdf["frac_in"] = stats_zonal.T
    gdf["geometry"] = aligned
    gdf["shift_m"] = np.hypot(gdf.dx_m, gdf.dy_m).round(2)
    gdf["qa_class"] = _classify_vec(gdf, o)
    for c in ("mean_prob", "cover", "frac_in", "dx_m", "dy_m"):
        gdf[c] = gdf[c].round(3)

    # --- 4. planarize the retained set ---
    clean = gdf[gdf.qa_class.isin(["keep", "review", "edge"])].copy()
    if o["planarize"] and len(clean):
        progress("Resolving overlaps (planarize)", 0.88)
        clean, ps = fx.planarize(clean)
    else:
        clean["area_m2"] = clean.geometry.area.round(1)
        ps = dict(n_clipped=0, n_emptied=0, overlap_removed=0.0, residual_m2=0.0,
                  sum_before=float(clean.geometry.area.sum()) if len(clean) else 0.0)

    # --- 4b. auto-clean slivers (policy) + QGIS-style error-point layer ---
    progress("Cleaning slivers + detecting topology errors", 0.94)
    clean, removed, rlog = tt.repair_topology(clean, policy=o["topo_policy"])
    # topology_removed = slivers (repair) + heavy-clip duplicates (planarize)
    import pandas as pd
    clip_removed = ps.get("clip_removed")
    if clip_removed is not None and len(clip_removed):
        removed = gpd.GeoDataFrame(pd.concat([removed, clip_removed], ignore_index=True),
                                   crs=crs)
    errors = tt.detect_topology_errors(
        gpd.GeoDataFrame(geometry=orig_geoms, crs=crs))

    counts = gdf.qa_class.value_counts().to_dict()
    stats = dict(
        n_in=n_in, n_iters=n_iters,
        med_shift_m=float(np.median(gdf.shift_m)) if n_in else 0.0,
        med_grid_m=float(np.median(gdf.grid_m)) if n_in else 0.0,
        n_refined=int(applied.sum()),
        counts=counts, n_clean=len(clean),
        planar_clipped=ps.get("n_clipped", 0),
        planar_emptied=ps.get("n_emptied", 0),
        heavy_clip_removed=ps.get("n_clip_removed", 0),
        overlap_removed_m2=round(ps.get("overlap_removed", 0.0), 1),
        residual_overlap_m2=round(ps.get("residual_m2", 0.0), 1),
        topo_policy=rlog["policy"], slivers_removed=rlog["n_removed"],
        slivers_removed_with_evidence=rlog["n_removed_with_evidence"],
        topo_errors=len(errors), topo_errors_by=errors.error_type.value_counts().to_dict()
        if len(errors) else {},
        seconds=round(time.time() - t0, 1),
    )
    original = gpd.GeoDataFrame(gdf.drop(columns="geometry"),
                                geometry=orig_geoms, crs=crs)
    progress("Done", 1.0)
    return dict(qa=gdf, clean=clean, removed=removed, errors=errors,
                original=original, img=img, prob=prob,
                transform=transform, crs=crs, bounds=bounds, stem=stem,
                prob_path=prob_path, stats=stats, options=o)


def prepare_fiftyone_package(gpkg, raster_path, out_dir, layer="agt_qa", chip=512):
    """Chip the raster over the polygon extent and write chips + a manifest.json
    of normalised polylines (label=qa_class). Done here in the geo-capable env so
    the FiftyOne launcher (separate 3.11 venv) only needs fiftyone + PIL.
    Returns the manifest path."""
    import json
    from PIL import Image
    from rasterio.windows import Window

    gdf = gpd.read_file(gpkg, layer=layer)
    with rasterio.open(raster_path) as r:
        gdf = gdf.to_crs(r.crs)
        inv = ~r.transform
        H, W = r.height, r.width
        pkg = os.path.join(out_dir, "fiftyone_chips")
        os.makedirs(pkg, exist_ok=True)
        # polygon extent in pixels
        minx, miny, maxx, maxy = gdf.total_bounds
        (c0, r0), (c1, r1) = inv * (minx, maxy), inv * (maxx, miny)
        cmin, cmax = max(int(min(c0, c1)), 0), min(int(max(c0, c1)) + 1, W)
        rmin, rmax = max(int(min(r0, r1)), 0), min(int(max(r0, r1)) + 1, H)
        geoms = list(gdf.geometry)
        labels = gdf["qa_class"].tolist() if "qa_class" in gdf else ["keep"] * len(gdf)
        tree = __import__("shapely").strtree.STRtree(geoms)
        samples = []
        for y in range(rmin, rmax, chip):
            for x in range(cmin, cmax, chip):
                w = min(chip, W - x)
                h = min(chip, H - y)
                if w < 8 or h < 8:
                    continue
                l, b, rt, t = array_bounds(h, w, r.window_transform(Window(x, y, w, h)))
                from shapely.geometry import box as _box
                cbox = _box(l, b, rt, t)
                idx = [i for i in tree.query(cbox) if geoms[i].intersects(cbox)]
                if not idx:
                    continue
                arr = r.read([1, 2, 3], window=Window(x, y, w, h))
                fp = os.path.join(pkg, f"chip_{y}_{x}.png")
                Image.fromarray(arr.transpose(1, 2, 0)).save(fp)
                polylines = []
                for i in idx:
                    clipped = geoms[i].intersection(cbox)
                    parts = getattr(clipped, "geoms", [clipped])
                    for p in parts:
                        if p.geom_type != "Polygon" or p.area < 0.25:
                            continue
                        xs, ys = p.exterior.coords.xy
                        pts = []
                        for gx, gy in zip(xs, ys):
                            px, py = inv * (gx, gy)
                            pts.append([(px - x) / w, (py - y) / h])
                        if len(pts) < 4:
                            continue
                        polylines.append({"label": labels[i], "points": [pts]})
                samples.append({"filepath": fp, "width": w, "height": h,
                                "polylines": polylines})
        manifest = os.path.join(pkg, "manifest.json")
        with open(manifest, "w") as fh:
            json.dump({"raster": raster_path, "n_samples": len(samples),
                       "samples": samples}, fh)
    return manifest


def save_outputs(result, out_dir, stem=None):
    """Write agt_qa / agt_clean / agt_original (+ topology_removed / topology_errors)
    layers to a gpkg; return path."""
    os.makedirs(out_dir, exist_ok=True)
    stem = stem or result["stem"]
    gpkg = os.path.join(out_dir, f"agt_fixed_{stem}.gpkg")
    if os.path.exists(gpkg):
        os.remove(gpkg)
    # Open-Buildings-style confidence on every fixed polygon (0-1 + high/medium/low band)
    try:
        import confidence_score as cs
        for k in ("qa", "clean"):
            if result.get(k) is not None and len(result[k]):
                result[k] = cs.add_confidence(result[k])
        result["confidence_summary"] = cs.summarise(result["clean"])
    except Exception as e:                      # never fail a run over the score
        print(f"  (confidence scoring skipped: {type(e).__name__}: {e})")
    result["qa"].to_file(gpkg, driver="GPKG", layer="agt_qa")
    result["clean"].to_file(gpkg, driver="GPKG", layer="agt_clean", mode="a")
    result["original"].to_file(gpkg, driver="GPKG", layer="agt_original", mode="a")
    if result.get("removed") is not None and len(result["removed"]):
        result["removed"].to_file(gpkg, driver="GPKG", layer="topology_removed", mode="a")
    if result.get("errors") is not None and len(result["errors"]):
        result["errors"].to_file(gpkg, driver="GPKG", layer="topology_errors", mode="a")
    return gpkg
