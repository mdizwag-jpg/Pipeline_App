# Shared building-footprint regularization library (v2, validated on
# Luanda_Informal_1 2026-07-23). Import-safe: functions only.
#   regularize(poly, parent_angle) -> (polygon, method)
#   postprocess(pieces_gdf, raw_gdf) -> final GeoDataFrame
# Pipeline per piece: confidence/area filter -> orientation-harmonized
# regularization (rect | ortho | simplified) -> clip to mask buffer ->
# priority overlap resolution (zero overlap by construction).
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, box as _box
from shapely.affinity import rotate, scale as _scale
from shapely import STRtree

CONF_MIN = 0.80     # keep detections at/above this confidence
MIN_AREA = 5.0      # m2
RECT_THR = 0.70     # rectangularity above this -> rectangle snap
SIMP = 0.5          # m, Douglas-Peucker before orthogonalization
ANG_TOL = 35.0      # deg, edges within this of 0/90 snap orthogonal
AREA_GUARD = (0.6, 1.45)
CLIP_M = 1.5        # regularized shape stays within this of its mask
ANGLE_SNAP = 25.0   # adopt parent roof-block angle if within this of own


def main_angle(poly):
    mrr = poly.minimum_rotated_rectangle
    c = np.asarray(mrr.exterior.coords)
    e = np.diff(c[:5], axis=0)
    v = e[np.argmax(np.hypot(e[:, 0], e[:, 1]))]
    return float(np.degrees(np.arctan2(v[1], v[0])))


def fixed_angle_rect(poly, theta):
    """Axis-aligned envelope in the frame rotated by theta, area-preserved."""
    org = poly.centroid
    r = rotate(poly, -theta, origin=org)
    env = _box(*r.bounds)
    rect_f = r.area / max(env.area, 1e-9)
    f = rect_f ** 0.5
    return rotate(_scale(env, xfact=f, yfact=f, origin="centroid"),
                  theta, origin=org), rect_f


def orthogonalize(poly, theta=None):
    """Snap edges to the dominant axis; returns None on failure."""
    if theta is None:
        theta = main_angle(poly)
    org = poly.centroid
    rot = rotate(poly, -theta, origin=org)
    ring = np.asarray(rot.exterior.simplify(SIMP).coords)[:-1]
    n = len(ring)
    if n < 3:
        return None

    dirs = []
    for i in range(n):
        v = ring[(i + 1) % n] - ring[i]
        ang = np.degrees(np.arctan2(v[1], v[0])) % 180.0
        if ang <= ANG_TOL or ang >= 180 - ANG_TOL:
            dirs.append(0)
        elif abs(ang - 90) <= ANG_TOL:
            dirs.append(90)
        else:
            dirs.append(None)

    start = next((i for i in range(n) if dirs[i] != dirs[i - 1]), 0)
    ring = np.roll(ring, -start, axis=0)
    dirs = dirs[start:] + dirs[:start]

    runs = []
    for i in range(n):
        p, q = ring[i], ring[(i + 1) % n]
        if runs and runs[-1][0] == dirs[i]:
            runs[-1][1].append((p, q))
        else:
            runs.append((dirs[i], [(p, q)]))
    if len(runs) >= 2 and runs[0][0] == runs[-1][0]:
        runs[-1][1].extend(runs[0][1]); runs.pop(0)
    if len(runs) < 4:
        return None

    lines = []
    for d, segs in runs:
        pts = np.array([pt for s in segs for pt in s])
        w = np.array([np.linalg.norm(q - p) for p, q in segs]).repeat(2)
        if d == 0:
            lines.append((0.0, 1.0, float(np.average(pts[:, 1], weights=w))))
        elif d == 90:
            lines.append((1.0, 0.0, float(np.average(pts[:, 0], weights=w))))
        else:
            p0, q1 = segs[0][0], segs[-1][1]
            dx, dy = q1 - p0
            norm = np.hypot(dx, dy)
            if norm < 1e-9:
                return None
            a, b = dy / norm, -dx / norm
            lines.append((a, b, a * p0[0] + b * p0[1]))

    m = len(lines)
    corners = []
    for i in range(m):
        a1, b1, c1 = lines[i]
        a2, b2, c2 = lines[(i + 1) % m]
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-6:
            return None
        corners.append(((c1 * b2 - c2 * b1) / det, (a1 * c2 - a2 * c1) / det))

    out = Polygon(corners)
    if not out.is_valid:
        out = out.buffer(0)
    if out.is_empty or out.geom_type != "Polygon":
        return None
    return rotate(out, theta, origin=org)


def regularize(poly, parent_angle=None):
    own_angle = main_angle(poly)
    theta = own_angle
    if parent_angle is not None:
        d = abs((own_angle - parent_angle + 90) % 180 - 90)
        if d <= ANGLE_SNAP:
            theta = parent_angle
    mrr = poly.minimum_rotated_rectangle
    rect_own = poly.area / max(mrr.area, 1e-9)
    aligned, rect_fixed = fixed_angle_rect(poly, theta)
    if rect_fixed >= RECT_THR - 0.05:
        return aligned, "rect"
    if rect_own >= RECT_THR:
        f = rect_own ** 0.5
        return _scale(mrr, xfact=f, yfact=f, origin="centroid"), "rect"
    reg = orthogonalize(poly, theta)
    if reg is not None:
        ratio = reg.area / max(poly.area, 1e-9)
        if AREA_GUARD[0] <= ratio <= AREA_GUARD[1] and reg.is_valid:
            return reg, "ortho"
    return poly.simplify(SIMP), "simplified"


def postprocess(pieces, raw, verbose=True):
    """pieces: GeoDataFrame with confidence + parent (index into raw).
    raw: GeoDataFrame of parent blobs (for orientation harmonization).
    Returns the final regularized, overlap-free GeoDataFrame."""
    keep = pieces[(pieces.confidence >= CONF_MIN)
                  & (pieces.geometry.area >= MIN_AREA)].copy()
    if verbose:
        print(f"pieces: {len(pieces)} -> {len(keep)} after conf/area filter")

    parent_angle = {}
    geoms, methods = [], []
    for g, par in zip(keep.geometry, keep.get("parent", [None] * len(keep))):
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda p: p.area)
        piece = Polygon(g.exterior)
        pa = None
        if par is not None and par == par:
            if par not in parent_angle:
                parent_angle[par] = main_angle(raw.geometry[par])
            pa = parent_angle[par]
        rg, how = regularize(piece, parent_angle=pa)
        rg = rg.intersection(piece.buffer(CLIP_M, join_style=2))
        if rg.geom_type == "MultiPolygon" and not rg.is_empty:
            rg = max(rg.geoms, key=lambda p: p.area)
        if rg.is_empty or rg.geom_type != "Polygon":
            rg, how = piece.simplify(SIMP), "simplified"
        geoms.append(rg); methods.append(how)

    final = gpd.GeoDataFrame(
        {"confidence": keep.confidence.round(3).values, "method": methods},
        geometry=geoms, crs=pieces.crs)
    final["area_m2"] = final.geometry.area
    final = final[final.area_m2 >= MIN_AREA].reset_index(drop=True)

    overlap_before = final.area_m2.sum() - final.geometry.union_all().area
    final = final.sort_values(["confidence", "area_m2"],
                              ascending=False).reset_index(drop=True)
    glist = list(final.geometry)
    tree = STRtree(glist)
    for i in range(len(glist)):
        higher = [j for j in tree.query(glist[i]) if j < i]
        g = glist[i]
        for j in sorted(higher):
            if not g.is_empty and g.intersects(glist[j]):
                g = g.difference(glist[j])
        glist[i] = g
    final["geometry"] = glist
    final = final[~final.geometry.is_empty].explode(index_parts=False)
    final = final[final.geometry.geom_type == "Polygon"]
    final["geometry"] = final.geometry.simplify(0.15)
    final["area_m2"] = final.geometry.area.round(1)
    final = final[final.area_m2 >= MIN_AREA].reset_index(drop=True)
    if verbose:
        overlap_after = final.area_m2.sum() - final.geometry.union_all().area
        print(f"overlap: {overlap_before/1000:.1f}k m2 -> {overlap_after:.1f} m2")
    return final
