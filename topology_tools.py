"""Topology error DETECTION (QGIS-style error-point layer) + automated REPAIR.

detect_topology_errors(gdf) -> a point GeoDataFrame with one marker at the
location of each problem (overlap region, sliver, invalid vertex, ...), like
QGIS Check Validity / Topology Checker.

repair_topology(gdf, policy) -> removes the deletion-class errors (tiny / thin
slivers) under a chosen policy and returns (kept, removed, log). Overlaps and
duplicates are resolved elsewhere in the pipeline (planarize / flag_duplicates);
this stage handles the sliver/fragment class that those don't.

Policies (exposed as checkboxes in the app):
  "aggressive" - remove every flagged sliver/fragment (max automation)
  "gated"      - remove only those that ALSO lack WV3 evidence (safe default)
  "none"       - remove nothing; flag for review only (geometry-only cleaning)
"""
import re

import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from shapely.strtree import STRtree
from shapely.validation import explain_validity

OVERLAP_EPS = 0.5      # m2 - ignore intersections below this (shared-edge noise)
SLIVER_AREA = 10.0     # m2 - tiny-area sliver
THIN_RATIO = 0.15      # 4*pi*A/P^2 below this = thin/spiky (1.0 = circle)
THIN_MAX_AREA = 10.0   # only auto-remove a thin polygon if also below this area
                       # (10 m2 = same ceiling as tiny; nothing larger is auto-deleted)
DUP_IOU = 0.98         # near-identical geometry

# NB: avoid the name "fid" - GeoPackage reserves it as the feature-id primary key,
# and our markers repeat source ids across error rows (UNIQUE-constraint clash).
_ERR_COLS = ["error_type", "severity", "value", "src_fid", "src_fid_other", "note", "geometry"]


def thinness(g):
    p = g.length
    return float(4 * np.pi * g.area / (p * p)) if p > 0 else 0.0


def _invalid_point(geom):
    """shapely reports the offending coordinate in the validity message, e.g.
    'Self-intersection[313456.2 -988123.4]' - place the marker exactly there."""
    msg = explain_validity(geom)
    m = re.search(r"\[\s*([-\d.eE]+)\s+([-\d.eE]+)\s*\]", msg)
    if m:
        return Point(float(m.group(1)), float(m.group(2))), msg
    return None, msg


def detect_topology_errors(gdf, min_area=SLIVER_AREA, thin_ratio=THIN_RATIO,
                           overlap_eps=OVERLAP_EPS, dup_iou=DUP_IOU):
    """Return a point layer of topology/validity errors, one marker per error."""
    fids = list(gdf.index)
    recs = []

    # --- validity (markers on the exact self-intersection vertex) ---
    gv = []
    for fid, g in zip(fids, gdf.geometry):
        if g is None or g.is_empty:
            gv.append(None)
            continue
        if not g.is_valid:
            pt, reason = _invalid_point(g)
            fixed = g.buffer(0)
            recs.append(dict(error_type="invalid", severity="error", value=np.nan,
                             src_fid=fid, src_fid_other=-1, note=reason.split("[")[0].strip(),
                             geometry=pt if pt is not None else fixed.representative_point()))
            gv.append(fixed)
        else:
            gv.append(g)

    idx = [k for k, g in enumerate(gv) if g is not None]
    geoms = [gv[k] for k in idx]
    areas = np.array([g.area for g in geoms])
    tree = STRtree(geoms)

    # --- overlaps / containment / near-duplicates ---
    for a, g in enumerate(geoms):
        for b in tree.query(g):
            if b <= a:
                continue
            inter = g.intersection(geoms[b])
            ia = inter.area
            if ia <= overlap_eps:
                continue
            i, j = idx[a], idx[b]
            iou = ia / (areas[a] + areas[b] - ia)
            rp = inter.representative_point()
            if iou >= dup_iou:
                recs.append(dict(error_type="duplicate", severity="error",
                                 value=round(iou, 3), src_fid=fids[i], src_fid_other=fids[j],
                                 note="near-identical", geometry=rp))
            elif ia / min(areas[a], areas[b]) > 0.9:
                small = i if areas[a] < areas[b] else j
                big = j if small == i else i
                recs.append(dict(error_type="contained", severity="error",
                                 value=round(ia, 1), src_fid=small, src_fid_other=big,
                                 note="inside larger polygon",
                                 geometry=geoms[a if small == i else b].representative_point()))
            else:
                recs.append(dict(error_type="overlap", severity="warning",
                                 value=round(ia, 1), src_fid=fids[i], src_fid_other=fids[j],
                                 note="partial overlap", geometry=rp))

    # --- slivers ---
    for a, g in enumerate(geoms):
        if areas[a] < min_area:
            recs.append(dict(error_type="tiny", severity="warning",
                             value=round(areas[a], 1), src_fid=fids[idx[a]], src_fid_other=-1,
                             note=f"area<{min_area:g} m2", geometry=g.representative_point()))
        elif thinness(g) < thin_ratio:
            recs.append(dict(error_type="thin", severity="warning",
                             value=round(thinness(g), 3), src_fid=fids[idx[a]], src_fid_other=-1,
                             note=f"compactness<{thin_ratio}", geometry=g.representative_point()))

    if not recs:
        return gpd.GeoDataFrame(columns=_ERR_COLS, geometry="geometry", crs=gdf.crs)
    return gpd.GeoDataFrame(recs, geometry="geometry", crs=gdf.crs)[_ERR_COLS]


def repair_topology(gdf, policy="aggressive", evidence_col="mean_prob",
                    evidence_min=0.15, min_area=SLIVER_AREA, thin_ratio=THIN_RATIO,
                    thin_max_area=THIN_MAX_AREA):
    """Remove sliver/fragment polygons per policy. Returns (kept, removed, log).
    Removed features are returned (not discarded) with a `remove_reason` so the
    caller can keep an audit layer."""
    g = gdf.copy()
    geoms = [x.buffer(0) if (x is not None and not x.is_valid) else x for x in g.geometry]
    areas = np.array([x.area if x is not None else 0.0 for x in geoms])
    thin = np.array([thinness(x) if x is not None else 1.0 for x in geoms])
    tiny = areas < min_area
    thin_small = (thin < thin_ratio) & (areas < thin_max_area)  # protect long real buildings
    candidate = tiny | thin_small
    reason = np.where(tiny, "tiny", np.where(thin_small, "thin", ""))

    if policy == "none":
        remove = np.zeros(len(g), bool)
    elif policy == "gated":
        ev = g[evidence_col].fillna(0).to_numpy() if evidence_col in g else np.zeros(len(g))
        remove = candidate & (ev < evidence_min)
    else:  # aggressive
        remove = candidate

    kept = g[~remove].copy()
    removed = g[remove].copy()
    removed["remove_reason"] = reason[remove]

    with_ev = 0
    if evidence_col in g:
        ev = g[evidence_col].fillna(0).to_numpy()
        with_ev = int((remove & (ev >= evidence_min)).sum())
    log = dict(policy=policy, n_candidates=int(candidate.sum()),
               n_removed=int(remove.sum()), n_kept=int((~remove).sum()),
               n_removed_with_evidence=with_ev)
    return kept, removed, log
