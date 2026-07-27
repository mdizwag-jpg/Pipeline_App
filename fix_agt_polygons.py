# Automatic QA/repair of the AGT (= Google Open Buildings v3) polygon layer
# against WV3 imagery, per clipped tile:
#   1. duplicate removal        - small polygons swallowed by a >=3x larger one,
#                                 and near-identical twins (IoU > 0.9)
#   2a. grid alignment          - displacement grid: per-cell template matching of
#                                 the rasterized polygon mask against the buildseg
#                                 building-probability raster, robust median +
#                                 interpolation, polygons translated per centroid.
#                                 Iterated to convergence: each pass re-measures the
#                                 RESIDUAL shift on the already-shifted mask and stops
#                                 when the median residual < CONV_PX. Offsets always
#                                 accumulate onto the ORIGINAL geometry (pure vector
#                                 translation, so no resampling drift).
#   2b. per-polygon refinement  - the smooth grid field helps the crowd but can push
#                                 an individual polygon off its roof (towers, locally
#                                 mis-digitized outlines). Each polygon is then
#                                 template-matched on its OWN footprint in a small
#                                 window and nudged, guarded by a neighbour-consensus
#                                 clamp so it can never fly onto a different building.
#   3. evidence scoring         - per-polygon mean probability + coverage under the
#                                 REFINED footprint -> keep / review / drop
#   4. topology / planarize     - resolve mutual overlaps in the retained set so no
#                                 area is shared/double-counted (higher-evidence
#                                 polygon wins the contested strip); union area is
#                                 preserved. Runs AFTER alignment so overlaps are
#                                 resolved on the corrected geometry.
#   4b. auto-clean + detect      - remove sliver/fragment polygons per TOPO_POLICY
#                                 (aggressive|gated|none) and emit a QGIS-style
#                                 error-point layer marking every detected topology
#                                 /validity problem in the delivered polygons.
# Layers written to agt_fixed_<tile>.gpkg:
#   agt_qa           - full record, aligned+scored+classified (unplanarized)
#   agt_clean        - planarized + de-slivered keep+review+edge deliverable
#   agt_original     - pre-shift geometry
#   topology_removed - polygons auto-removed by 4b (audit trail; reversible)
#   topology_errors  - point per detected error (overlap/contained/dup/tiny/thin/invalid)
# Usage (from Notebooks dir): python fix_agt_polygons.py <tile_name | all>
import glob
import os
import sys
import time

import cv2
import numpy as np
import geopandas as gpd
import pyogrio
import rasterio
import shapely
from pyproj import Transformer
from rasterio import windows as rio_windows
from rasterio.features import rasterize
from rasterio.transform import Affine, array_bounds
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from shapely.affinity import translate
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import topology_tools as tt

BASE = r"G:\My Drive\Innocom Global Geospatial Projects\Projects\Angola\Projects\Tax Recovery"
AGT = os.path.join(BASE, "Data", "Luanda_Polygons", "agt_luanda_full.gpkg")
OUTDIR = os.path.join(BASE, "Data", "Outputs", "agt_fix")

EXG_VEG = 30.0                  # same ExG threshold as the buildseg pipeline
CELL, STEP, MARGIN = 512, 256, 48   # displacement grid: cell px, stride, +-search px
MIN_TPX = 1500                  # template needs >= this many building px (~150 m2)
MIN_CORR = 0.12                 # reject weak correlation peaks
CONV_PX = 1.0                   # grid loop stops when median residual shift < this (px, ~0.31 m)
MAX_ITERS = 5                   # hard cap on grid-alignment iterations
REFINE_SEARCH_PX = 12           # per-polygon search half-window (~3.8 m) - small so a
                                # polygon cannot reach a different building
REFINE_MIN_DIMPX = 6            # skip refining polygons smaller than this in either axis
REFINE_MIN_TPX = 60             # ...or with fewer building px than this (too small to match)
REFINE_MINCORR = 0.25           # per-polygon peak must beat this (stricter than the grid)
REFINE_NEI_TOL = 8.0            # runaway guard: clamp a nudge to <= this px (~2.5 m) from
                                # the local neighbour-consensus offset
REFINE_MAXMOVE = 12.0           # absolute cap on a single nudge (px)
REFINE_ENABLED = False          # per-polygon refinement OFF by default — ground-truth IoU
                                # showed it hurts on informal fabric (chases blobby masks
                                # off true footprints). Grid alignment alone is safe.
KEEP_MEAN, KEEP_COVER = 0.40, 0.40
DROP_MEAN, DROP_COVER = 0.15, 0.15
# refinement is allowed ONLY where a building already exists near the delivered
# position (original OR grid), so it can never rescue a bare-ground false positive
REFINE_EVID_MEAN, REFINE_EVID_COVER = DROP_MEAN, DROP_COVER
EDGE_FRAC = 0.60                # < this fraction of the polygon inside the tile -> 'edge'
DUP_RATIO, DUP_OVER, TWIN_IOU = 3.0, 0.5, 0.9
PLANARIZE_MIN_PART = 1.0        # m2 - drop shards smaller than this after differencing
CLIP_REMOVE_FRAC = 0.6          # planarize: a polygon clipped to lose more than this
                                # fraction of its area is a near-duplicate of its
                                # neighbour -> removed (to topology_removed) not kept as a sliver
TOPO_POLICY = "gated"           # auto-clean policy: gated (default, keeps evidence-backed
                                # small buildings) | aggressive | none. See methodology report.

CLASS_COLORS = {"keep": "lime", "review": "orange", "drop": "red",
                "duplicate": "magenta", "edge": "deepskyblue"}


def prob_raster(tile_name, img_hwc, transform, crs):
    """buildseg probability, cached as GeoTIFF so reruns skip inference."""
    cache = os.path.join(OUTDIR, f"prob_{tile_name}.tif")
    if os.path.exists(cache):
        with rasterio.open(cache) as r:
            return r.read(1)
    import onnxruntime as ort
    import buildseg_pipeline_tile as bpt
    sess = ort.InferenceSession(bpt.ONNX, providers=["CPUExecutionProvider"])
    prob = bpt.bseg_prob(img_hwc, sess, sess.get_inputs()[0].name)
    with rasterio.open(cache, "w", driver="GTiff", width=prob.shape[1],
                       height=prob.shape[0], count=1, dtype="float32",
                       transform=transform, crs=crs, compress="lzw") as w:
        w.write(prob.astype("float32"), 1)
    return prob


def load_agt(bounds, crs):
    tr = Transformer.from_crs(crs, 4326, always_xy=True)
    l, b, r, t = bounds
    x0, y0 = tr.transform(l, b)
    x1, y1 = tr.transform(r, t)
    gdf = pyogrio.read_dataframe(AGT, bbox=(x0, y0, x1, y1))
    gdf = gdf.to_crs(crs)
    gdf["geometry"] = gdf.geometry.make_valid()
    gdf = gdf[~gdf.geometry.is_empty & (gdf.geometry.area > 0)].reset_index(drop=True)
    return gdf


def flag_duplicates(gdf):
    """'duplicate' = mostly covered by a >=DUP_RATIO x larger polygon, or a
    near-identical twin of an earlier row."""
    geoms = list(gdf.geometry)
    areas = np.array([g.area for g in geoms])
    tree = STRtree(geoms)
    dup = np.zeros(len(geoms), bool)
    for i, g in enumerate(geoms):
        cand = [j for j in tree.query(g) if j != i]
        over = 0.0
        for j in cand:
            inter = g.intersection(geoms[j]).area
            if inter <= 0:
                continue
            iou = inter / (areas[i] + areas[j] - inter)
            if iou > TWIN_IOU and j < i:          # exact twin: keep first row
                dup[i] = True
            if areas[j] >= DUP_RATIO * areas[i] and not dup[j]:
                over += inter
        if over / areas[i] > DUP_OVER:            # swallowed by larger neighbors
            dup[i] = True
    return dup


def displacement_grid(geoms, prob_eff, transform, shape):
    """Per-cell offset (px) of the polygon mask vs the probability raster."""
    H, W = shape
    maskf = rasterize(((g, 1) for g in geoms), out_shape=shape,
                      transform=transform, dtype="uint8").astype("float32")
    ch, cw = min(CELL, H), min(CELL, W)
    my = min(MARGIN, max((ch - 64) // 2, 1))
    mx = min(MARGIN, max((cw - 64) // 2, 1))
    ys = sorted({min(y, H - ch) for y in range(0, max(H - ch, 0) + 1, STEP)})
    xs = sorted({min(x, W - cw) for x in range(0, max(W - cw, 0) + 1, STEP)})
    cells = []
    for y in ys:
        for x in xs:
            S = prob_eff[y:y + ch, x:x + cw]
            T = maskf[y + my:y + ch - my, x + mx:x + cw - mx]
            if T.sum() < MIN_TPX or T.std() == 0 or S.std() == 0:
                continue
            res = cv2.matchTemplate(S, T, cv2.TM_CCOEFF_NORMED)
            _, maxv, _, maxloc = cv2.minMaxLoc(res)
            if maxv < MIN_CORR:
                continue
            cells.append((x + cw / 2, y + ch / 2,
                          maxloc[0] - mx, maxloc[1] - my, float(maxv)))
    if not cells:
        return None, []
    arr = np.array(cells)
    med = np.median(arr[:, 2:4], axis=0)
    mad = np.median(np.abs(arr[:, 2:4] - med), axis=0)
    tol = np.maximum(3 * 1.4826 * mad, 3.0)
    ok = np.all(np.abs(arr[:, 2:4] - med) <= tol, axis=1)
    return arr[ok], [tuple(c) for c in arr[~ok]]


def per_poly_offsets(cells, cents_px):
    """Interpolate cell offsets at polygon centroids (px coords)."""
    if len(cells) >= 4:
        vals = []
        for k in (2, 3):
            v = griddata(cells[:, :2], cells[:, k], cents_px, method="linear")
            vn = griddata(cells[:, :2], cells[:, k], cents_px, method="nearest")
            vals.append(np.where(np.isnan(v), vn, v))
        return np.column_stack(vals)
    med = np.median(cells[:, 2:4], axis=0)
    return np.tile(med, (len(cents_px), 1))


def refine_offsets(geoms, prob_eff, transform, eligible):
    """Per-polygon local nudge (px) on top of the grid alignment.

    Each polygon is template-matched on its OWN footprint within a small
    +-REFINE_SEARCH_PX window of its current position, then the raw nudge is
    clamped to REFINE_NEI_TOL px of its neighbours' consensus so a single
    polygon can never snap onto a different building. Polygons too small or
    too weakly correlated to match reliably keep their grid position (nudge 0).

    `eligible[i]` gates whether polygon i may be nudged at all. Only polygons
    with real evidence at their as-delivered/grid position are eligible; this
    stops the refinement from sliding a bare-ground false positive onto a
    neighbouring roof and thereby manufacturing a false keep (measured: without
    the gate, 25 of 34 drop-rescues on Informal_1 were spurious).
    Returns (refine_px[N,2], corr[N], applied_mask[N]).
    """
    H, W = prob_eff.shape
    N = len(geoms)
    inv = ~transform
    raw = np.full((N, 2), np.nan, "float64")
    corr = np.zeros(N)
    for i, g in enumerate(geoms):
        if not eligible[i]:
            continue
        minx, miny, maxx, maxy = g.bounds
        c0f, r0f = inv * (minx, maxy)      # top-left px (maxy -> smaller row)
        c1f, r1f = inv * (maxx, miny)
        c0, r0 = int(np.floor(c0f)), int(np.floor(r0f))
        c1, r1 = int(np.ceil(c1f)), int(np.ceil(r1f))
        th, tw = r1 - r0, c1 - c0
        if min(th, tw) < REFINE_MIN_DIMPX:
            continue
        m = REFINE_SEARCH_PX
        sr0, sc0 = max(r0 - m, 0), max(c0 - m, 0)
        sr1, sc1 = min(r1 + m, H), min(c1 + m, W)
        S = prob_eff[sr0:sr1, sc0:sc1]
        if S.shape[0] <= th or S.shape[1] <= tw or S.std() == 0:
            continue
        sub_tr = transform * Affine.translation(c0, r0)
        T = rasterize([(g, 1)], out_shape=(th, tw), transform=sub_tr,
                      dtype="uint8").astype("float32")
        if T.sum() < REFINE_MIN_TPX or T.std() == 0:
            continue
        res = cv2.matchTemplate(S, T, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxloc = cv2.minMaxLoc(res)
        if maxv < REFINE_MINCORR:
            continue
        raw[i] = (maxloc[0] - (c0 - sc0), maxloc[1] - (r0 - sr0))  # nudge from current pos
        corr[i] = maxv

    refine = np.zeros((N, 2))
    applied = np.zeros(N, bool)
    elig = np.where(np.isfinite(raw[:, 0]))[0]
    if len(elig) == 0:
        return refine, corr, applied
    cents = np.array([[g.centroid.x, g.centroid.y] for g in geoms])
    tree = cKDTree(cents[elig])
    k = min(9, len(elig))
    for i in elig:
        _, nbr = tree.query(cents[i], k=k)
        nei_med = np.median(raw[elig[np.atleast_1d(nbr)]], axis=0)
        off = raw[i].copy()
        dev = off - nei_med
        dn = np.hypot(*dev)
        if dn > REFINE_NEI_TOL:                    # runaway -> clamp toward consensus
            off = nei_med + dev / dn * REFINE_NEI_TOL
        mag = np.hypot(*off)
        if mag > REFINE_MAXMOVE:
            off = off / mag * REFINE_MAXMOVE
        refine[i] = off
        applied[i] = np.hypot(*off) > 1e-6
    return refine, corr, applied


def zonal(geom, prob_eff, transform, px_area):
    """(mean_prob, cover, frac_inside_tile) under the polygon."""
    H, W = prob_eff.shape
    win = rio_windows.from_bounds(*geom.bounds, transform=transform)
    win = win.round_offsets().round_lengths()
    r0, c0 = max(int(win.row_off), 0), max(int(win.col_off), 0)
    r1 = min(int(win.row_off + win.height) + 1, H)
    c1 = min(int(win.col_off + win.width) + 1, W)
    if r1 <= r0 or c1 <= c0:
        return np.nan, np.nan, 0.0
    sub_tr = transform * transform.__class__.translation(c0, r0)
    m = rasterize([(geom, 1)], out_shape=(r1 - r0, c1 - c0),
                  transform=sub_tr, dtype="uint8").astype(bool)
    n = int(m.sum())
    if n == 0:
        return np.nan, np.nan, 0.0
    p = prob_eff[r0:r1, c0:c1][m]
    return float(p.mean()), float((p > 0.5).mean()), n * px_area / geom.area


def classify(row):
    if row.qa_class == "duplicate":
        return "duplicate"
    if row.frac_in < EDGE_FRAC:
        return "edge"
    if row.mean_prob >= KEEP_MEAN or row.cover >= KEEP_COVER:
        return "keep"
    if row.mean_prob < DROP_MEAN and row.cover < DROP_COVER:
        return "drop"
    return "review"


def _clean_parts(geom):
    """Keep one feature per row: drop shards < PLANARIZE_MIN_PART, return a
    Polygon/MultiPolygon (or None if nothing meaningful survives)."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom if geom.area >= PLANARIZE_MIN_PART else None
    parts = [p for p in getattr(geom, "geoms", [])
             if p.geom_type == "Polygon" and p.area >= PLANARIZE_MIN_PART]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def planarize(gdf):
    """Resolve mutual overlaps so the retained layer is planar (no shared area).

    Each contested region is assigned to the higher-priority polygon
    (priority = evidence mean_prob desc, then area desc) by subtracting all
    higher-priority neighbours from every polygon. Any point ends up owned by
    exactly the highest-priority polygon covering it, so the union area is
    preserved and no area is double-counted -- what a summed tax roll needs.
    Returns (clean_gdf, stats).
    """
    g = gdf.reset_index(drop=True).copy()
    geoms = [gg.buffer(0) if not gg.is_valid else gg for gg in g.geometry]
    areas = np.array([x.area for x in geoms])
    mp = g["mean_prob"].fillna(0.0).to_numpy()
    order = np.lexsort((areas, mp))[::-1]          # highest priority first
    rank = np.empty(len(geoms), int)
    rank[order] = np.arange(len(geoms))
    tree = STRtree(geoms)
    sum_before = float(areas.sum())
    union_before = float(unary_union(geoms).area)

    new, clipped = [], 0
    clip_frac = np.zeros(len(geoms))
    for i, geom in enumerate(geoms):
        higher = [geoms[j] for j in tree.query(geom)
                  if j != i and rank[j] < rank[i]
                  and geom.intersection(geoms[j]).area > 1e-9]
        if higher:
            ng = geom.difference(unary_union(higher))
            clipped += ng.area < geom.area - 1e-6
            if areas[i] > 0:
                clip_frac[i] = max(0.0, 1.0 - ng.area / areas[i])
            new.append(ng)
        else:
            new.append(geom)

    cleaned = [_clean_parts(x) for x in new]
    valid = np.array([c is not None and not c.is_empty for c in cleaned])
    heavy = clip_frac > CLIP_REMOVE_FRAC           # near-duplicate of its neighbour
    keep = valid & ~heavy
    emptied = int((valid & heavy).sum() + (~valid).sum())  # all removed as heavy-clip dups

    g["clip_frac"] = clip_frac.round(3)
    kept = g[keep].copy()
    kept["geometry"] = [cleaned[i] for i in np.where(keep)[0]]
    kept["area_m2"] = kept.geometry.area.round(1)

    # heavy-clip / emptied losers -> removed as duplicates (keep their ALIGNED geom)
    rm = ~keep
    clip_removed = g[rm].copy()                     # aligned (pre-clip) geometry from gdf
    clip_removed["area_m2"] = clip_removed.geometry.area.round(1)
    clip_removed["remove_reason"] = "heavy_clip_duplicate"

    gg = list(kept.geometry)
    t2 = STRtree(gg)
    resid = sum(gg[i].intersection(gg[j]).area
                for i, geom in enumerate(gg) for j in t2.query(geom) if j > i)
    stats = dict(sum_before=sum_before, union_before=union_before,
                 sum_after=float(kept.area_m2.sum()) if len(kept) else 0.0,
                 n_clipped=int(clipped), n_emptied=emptied,
                 n_clip_removed=int(rm.sum()), clip_removed=clip_removed,
                 overlap_removed=sum_before - (float(kept.area_m2.sum()) if len(kept) else 0.0),
                 residual_m2=float(resid))
    return kept, stats


def main(tile_name):
    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    tif = os.path.join(BASE, "Data", "Clipped_Training", f"{tile_name}.tif")
    with rasterio.open(tif) as r:
        img_hwc = r.read([1, 2, 3]).transpose(1, 2, 0)
        transform, crs, bounds = r.transform, r.crs, r.bounds
        H, W = r.height, r.width
    px_area = abs(transform.a * transform.e)

    prob = prob_raster(tile_name, img_hwc, transform, crs)
    f = img_hwc.astype("float32")
    veg = (2 * f[:, :, 1] - f[:, :, 0] - f[:, :, 2]) > EXG_VEG
    prob_eff = np.where(veg, 0.0, prob).astype("float32")

    gdf = load_agt(bounds, crs)
    print(f"{tile_name}: {len(gdf)} AGT polygons, prob ready ({time.time()-t0:.0f}s)",
          flush=True)

    gdf["qa_class"] = ""
    dup = flag_duplicates(gdf)
    gdf.loc[dup, "qa_class"] = "duplicate"

    orig_geoms = list(gdf.geometry)
    inv = ~transform
    cents0 = np.array([inv * (g.centroid.x, g.centroid.y) for g in orig_geoms])

    # --- 2a. iterative grid alignment (accumulate px offset onto ORIGINAL geoms) ---
    cum = np.zeros((len(gdf), 2))           # cumulative grid offset, px, per polygon
    n_iters = 0
    for it in range(MAX_ITERS):
        shifted = [translate(g, xoff=cum[i, 0] * transform.a,
                             yoff=cum[i, 1] * transform.e)
                   for i, g in enumerate(orig_geoms)]
        live_geoms = [shifted[i] for i in range(len(gdf)) if not dup[i]]
        cells, rejected = displacement_grid(live_geoms, prob_eff, transform, (H, W))
        if cells is None or len(cells) == 0:
            if it == 0:
                print("  WARNING: no alignment cells accepted - polygons left unshifted")
            break
        n_iters = it + 1
        res_med = np.median(cells[:, 2:4], axis=0)
        step = per_poly_offsets(cells, cents0 + cum)
        cum += step
        rmag = float(np.hypot(*res_med))
        print(f"  grid iter {n_iters}: {len(cells)} cells / {len(rejected)} rejected | "
              f"residual px=({res_med[0]:+.1f},{res_med[1]:+.1f}) "
              f"= {rmag*abs(transform.a):.2f} m | cum med "
              f"{np.median(np.hypot(cum[:,0],cum[:,1]))*abs(transform.a):.2f} m", flush=True)
        if rmag < CONV_PX:
            break

    grid_geoms = [translate(g, xoff=cum[i, 0] * transform.a,
                            yoff=cum[i, 1] * transform.e)
                  for i, g in enumerate(orig_geoms)]

    # --- honest (non-circular) evidence: at the AS-DELIVERED and GRID positions.
    # Refinement (per-polygon snapping) optimizes overlap with the same raster the
    # scoring reads, so its evidence is circular and must NOT gate keep/drop.
    gm = np.array([zonal(g, prob_eff, transform, px_area) for g in grid_geoms])
    # --- 2b. per-polygon refinement — OFF by default. Ground-truth IoU showed it
    # DEGRADES accuracy on informal fabric (it chases the blobby buildseg mask off
    # the true footprint). Grid alignment alone is safe. Toggle via REFINE_ENABLED.
    refine = np.zeros((len(gdf), 2))
    applied = np.zeros(len(gdf), bool)
    if REFINE_ENABLED:
        om = np.array([zonal(g, prob_eff, transform, px_area) for g in orig_geoms])
        building = ((np.nan_to_num(om[:, 0]) >= REFINE_EVID_MEAN) |
                    (np.nan_to_num(om[:, 1]) >= REFINE_EVID_COVER) |
                    (np.nan_to_num(gm[:, 0]) >= REFINE_EVID_MEAN) |
                    (np.nan_to_num(gm[:, 1]) >= REFINE_EVID_COVER))
        eligible = building & ~dup
        refine, rcorr, applied = refine_offsets(grid_geoms, prob_eff, transform, eligible)
        print(f"  refine: {int(applied.sum())} of {int(eligible.sum())} eligible nudged",
              flush=True)
    total = cum + refine

    gdf["dx_m"] = total[:, 0] * transform.a
    gdf["dy_m"] = total[:, 1] * transform.e
    gdf["grid_m"] = np.hypot(cum[:, 0] * transform.a, cum[:, 1] * transform.e)
    gdf["refine_m"] = np.hypot(refine[:, 0] * transform.a, refine[:, 1] * transform.e)
    gdf["refined"] = applied
    gdf["grid_mean"] = np.round(np.nan_to_num(gm[:, 0]), 3)   # honest pre-snap evidence
    aligned = [translate(g, xoff=dx, yoff=dy) for g, dx, dy
               in zip(orig_geoms, gdf.dx_m, gdf.dy_m)]

    stats = np.array([zonal(g, prob_eff, transform, px_area) for g in aligned])
    gdf["mean_prob"], gdf["cover"], gdf["frac_in"] = stats.T
    gdf["geometry"] = aligned
    gdf["shift_m"] = np.hypot(gdf.dx_m, gdf.dy_m).round(2)
    gdf["qa_class"] = gdf.apply(classify, axis=1)
    for c in ("mean_prob", "cover", "frac_in", "dx_m", "dy_m", "grid_m", "refine_m"):
        gdf[c] = gdf[c].round(3)

    counts = gdf.qa_class.value_counts().to_dict()
    print(f"  classes: {counts}", flush=True)

    # --- 4. topology: planarize the retained set (no shared/double-counted area) ---
    clean = gdf[gdf.qa_class.isin(["keep", "review", "edge"])].copy()
    clean_planar, ps = planarize(clean)
    print(f"  planarize: {ps['n_clipped']} clipped, {ps['n_clip_removed']} removed as "
          f"heavy-clip duplicates (>{int(CLIP_REMOVE_FRAC*100)}% lost) | "
          f"overlap removed {ps['overlap_removed']:.0f} m2 "
          f"({100*ps['overlap_removed']/ps['sum_before']:.2f}% of clean area) | "
          f"residual overlap {ps['residual_m2']:.1f} m2", flush=True)

    # --- 4b. automated sliver/fragment removal (policy-driven) + error-point layer ---
    clean_kept, removed, rlog = tt.repair_topology(clean_planar, policy=TOPO_POLICY)
    # topology_removed = slivers (repair) + heavy-clip duplicates (planarize)
    import pandas as pd
    removed = gpd.GeoDataFrame(
        pd.concat([removed, ps["clip_removed"]], ignore_index=True), crs=crs) \
        if len(ps["clip_removed"]) else removed
    print(f"  auto-clean [{rlog['policy']}]: removed {rlog['n_removed']} sliver/fragment "
          f"of {rlog['n_candidates']} flagged"
          + (f" ({rlog['n_removed_with_evidence']} of them had building evidence — "
             f"would survive 'gated')" if rlog['n_removed_with_evidence'] else ""), flush=True)
    errs = tt.detect_topology_errors(
        gpd.GeoDataFrame(geometry=orig_geoms, crs=crs))   # audit of delivered polygons
    print(f"  topology errors detected in delivered layer: {len(errs)}"
          + (f" — {dict(errs.error_type.value_counts())}" if len(errs) else ""), flush=True)

    # Open-Buildings-style per-polygon confidence (0-1 + high/medium/low band) so every
    # fixed polygon carries one interpretable score rather than five raw diagnostics.
    import confidence_score as cs
    gdf = cs.add_confidence(gdf)
    clean_kept = cs.add_confidence(clean_kept)
    _cs = cs.summarise(clean_kept)
    if _cs:
        print(f"  confidence (deliverable): median {_cs['median']:.2f} | "
              f"high {_cs['high']} ({_cs['high_pct']}%) / medium {_cs['medium']} "
              f"({_cs['medium_pct']}%) / low {_cs['low']} ({_cs['low_pct']}%)", flush=True)

    out_gpkg = os.path.join(OUTDIR, f"agt_fixed_{tile_name}.gpkg")
    if os.path.exists(out_gpkg):
        os.remove(out_gpkg)
    gdf.to_file(out_gpkg, driver="GPKG", layer="agt_qa")          # full record (unplanarized)
    clean_kept.to_file(out_gpkg, driver="GPKG", layer="agt_clean", mode="a")  # planar + de-slivered
    gpd.GeoDataFrame(gdf.drop(columns="geometry"), geometry=orig_geoms,
                     crs=crs).to_file(out_gpkg, driver="GPKG",
                                      layer="agt_original", mode="a")
    if len(removed):
        removed.to_file(out_gpkg, driver="GPKG", layer="topology_removed", mode="a")
    if len(errs):
        errs.to_file(out_gpkg, driver="GPKG", layer="topology_errors", mode="a")

    l, b, rt, t = array_bounds(H, W, transform)
    fig, axes = plt.subplots(2, 1, figsize=(14, 15), dpi=140)
    axes[0].imshow(img_hwc, extent=(l, rt, b, t))
    gpd.GeoSeries(orig_geoms, crs=crs).boundary.plot(ax=axes[0], color="yellow",
                                                     linewidth=0.7)
    axes[0].set_title(f"{tile_name} BEFORE: {len(gdf)} AGT polygons as delivered")
    axes[1].imshow(img_hwc, extent=(l, rt, b, t))
    # AFTER shows the DELIVERABLE: kept classes from the planarized clean_kept
    # (so no shared/overlapping area), plus removed classes (drop/duplicate) from
    # gdf for context. Plotting gdf here would show the pre-planarize overlaps.
    for cls in ("keep", "review", "edge"):
        sub = clean_kept[clean_kept.qa_class == cls] if "qa_class" in clean_kept else clean_kept.iloc[0:0]
        if len(sub):
            sub.boundary.plot(ax=axes[1], color=CLASS_COLORS[cls], linewidth=0.7,
                              label=f"{cls} ({len(sub)})")
    for cls in ("drop", "duplicate"):
        sub = gdf[gdf.qa_class == cls]
        if len(sub):
            sub.boundary.plot(ax=axes[1], color=CLASS_COLORS[cls], linewidth=0.7,
                              label=f"{cls} ({len(sub)})")
    axes[1].legend(loc="lower right", fontsize=9)
    axes[1].set_title(f"{tile_name} AFTER: planarized deliverable "
                      f"({len(clean_kept)} clean, 0 overlap) + removed "
                      f"(median shift {np.median(gdf.shift_m):.1f} m)")
    for ax in axes:
        ax.set_axis_off()
    fig.tight_layout()
    out_png = os.path.join(OUTDIR, f"agt_fixed_{tile_name}_overlay.png")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    summary = dict(tile=tile_name, n_in=len(gdf), n_iters=n_iters,
                   med_shift_m=float(np.median(gdf.shift_m)),
                   med_grid_m=float(np.median(gdf.grid_m)),
                   n_refined=int(gdf.refined.sum()),
                   med_refine_m=float(gdf.loc[gdf.refined, "refine_m"].median())
                   if gdf.refined.any() else 0.0,
                   **{f"n_{k}": counts.get(k, 0) for k in CLASS_COLORS},
                   n_clean=len(clean_kept), planar_clipped=ps["n_clipped"],
                   heavy_clip_removed=ps["n_clip_removed"],
                   overlap_removed_m2=round(ps["overlap_removed"], 1),
                   residual_overlap_m2=round(ps["residual_m2"], 1),
                   topo_policy=rlog["policy"], slivers_removed=rlog["n_removed"],
                   slivers_removed_with_evidence=rlog["n_removed_with_evidence"],
                   topo_errors=len(errs))
    print(f"  -> {out_gpkg}\n  -> {out_png}  ({time.time()-t0:.0f}s)", flush=True)
    return summary


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "all":
        tiles = [os.path.splitext(os.path.basename(p))[0] for p in
                 sorted(glob.glob(os.path.join(BASE, "Data", "Clipped_Training", "*.tif")))
                 if "mask" not in p]
    else:
        tiles = [arg]
    rows = [main(t) for t in tiles]
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, "summary.csv"), index=False)
    print("\n" + df.to_string(index=False))
