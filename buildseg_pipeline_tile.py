# End-to-end buildseg final-product pipeline for one clipped tile:
#   ONNX sliding-window inference -> mask + ExG veg screen -> blob confidence
#   -> ws_dist instance split -> regularize_lib.postprocess (v2)
# Usage: python buildseg_pipeline_tile.py Luanda_formal_1
# Outputs: Data/Outputs/buildseg_final_<tile>.gpkg + overlay PNG.
import os, sys, time
import numpy as np
import rasterio
from rasterio.features import shapes as rio_shapes
from rasterio.transform import array_bounds
from scipy import ndimage as ndi
import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import unary_union
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.filters import sobel, gaussian
import cv2
import onnxruntime as ort

import regularize_lib as rl

BASE = r"G:\My Drive\Innocom Global Geospatial Projects\Projects\Angola\Projects\Tax Recovery"
ONNX = os.path.join(BASE, "Data", "Models", "buildseg_segformer_b2.onnx")

TILE, OVERLAP = 512, 128
STRIDE = TILE - OVERLAP
MIN_PX = 20
EXG_VEG = 30.0
MIN_DIST_PX = 8     # ~2.5 m between roof-center watershed seeds
SEED_MIN_D = 2.0    # seed only where >=2 px inside the mask
BIG_M2 = 250.0      # cascade: re-split pieces bigger than this...
RECT_TH = 0.6       # ...that are also less rectangular than this


def seam_energy(img_hwc):
    """Dark alley/eave seams + edges — watershed cuts follow visible roof
    boundaries (validated best splitter outside Q1)."""
    gray = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2GRAY).astype("float32") / 255.0
    s = gaussian(1.0 - gray, 1.0) + 2.0 * gaussian(sobel(gray), 1.0)
    return (s / max(s.max(), 1e-6)).astype("float32")


def rectangularity_px(m):
    cnts, _ = cv2.findContours(m.astype("uint8"), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 1.0
    rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    ra = rect[1][0] * rect[1][1]
    return float(m.sum()) / ra if ra > 0 else 1.0


def cascade(lab, energy, px_m2, md=MIN_DIST_PX):
    """Re-split pieces > BIG_M2 with rectangularity < RECT_TH (protects
    genuine large rectangular buildings). Adopted production behavior."""
    out = lab.copy()
    nxt = int(lab.max()) + 1
    ids, counts = np.unique(lab[lab > 0], return_counts=True)
    slices = ndi.find_objects(lab)
    for iid, cnt in zip(ids, counts):
        if cnt * px_m2 < BIG_M2:
            continue
        sl = slices[iid - 1]
        if sl is None:
            continue
        m = lab[sl] == iid
        if rectangularity_px(m) >= RECT_TH:
            continue
        D = ndi.distance_transform_edt(m)
        pk = peak_local_max(D, min_distance=md, threshold_abs=2.0,
                            exclude_border=False, labels=m)
        if len(pk) < 2:
            continue
        seeds = np.zeros(m.shape, bool); seeds[tuple(pk.T)] = True
        mk, _ = ndi.label(seeds)
        sub = watershed(energy[sl], mk, mask=m)
        piece = out[sl]
        piece[m] = sub[m] + nxt
        nxt += int(sub.max()) + 1
    u = np.unique(out)
    remap = np.zeros(u.max() + 1, "int32"); remap[u] = np.arange(len(u))
    return remap[out]


def bseg_prob(img_hwc, sess, iname):
    H0, W0 = img_hwc.shape[:2]
    if H0 < TILE or W0 < TILE:   # pad small tiles up to one window
        img_hwc = np.pad(img_hwc, ((0, max(0, TILE - H0)),
                                   (0, max(0, TILE - W0)), (0, 0)), mode="reflect")
    H, W = img_hwc.shape[:2]
    w1 = np.hanning(TILE + 2)[1:-1].astype("float32")
    win2d = np.clip(np.outer(w1, w1), 1e-3, None)
    acc = np.zeros((H, W), "float32"); wgt = np.zeros((H, W), "float32")
    ys = sorted({min(y, H - TILE) for y in range(0, max(H - TILE, 0) + 1, STRIDE)} | {max(H - TILE, 0)})
    xs = sorted({min(x, W - TILE) for x in range(0, max(W - TILE, 0) + 1, STRIDE)} | {max(W - TILE, 0)})
    n = len(ys) * len(xs); done = 0
    for y in ys:
        for x in xs:
            chip = img_hwc[y:y + TILE, x:x + TILE].astype("float32")
            xin = ((chip / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None].astype("float32")
            r = sess.run(None, {iname: xin})[0][0]
            e = np.exp(r - r.max(0, keepdims=True))
            acc[y:y + TILE, x:x + TILE] += (e / e.sum(0, keepdims=True))[1] * win2d
            wgt[y:y + TILE, x:x + TILE] += win2d
            done += 1
            if done % 10 == 0 or done == n:
                print(f"  window {done}/{n}", flush=True)
    return (acc / wgt)[:H0, :W0]


def polygonize(lab, transform, values=None):
    geoms_by_id = {}
    for g, v in rio_shapes(lab.astype("int32"), mask=lab > 0, transform=transform):
        geoms_by_id.setdefault(int(v), []).append(shape(g))
    ids = sorted(geoms_by_id)
    geoms = [unary_union(geoms_by_id[i]) if len(geoms_by_id[i]) > 1
             else geoms_by_id[i][0] for i in ids]
    return ids, geoms


def main(tile_name):
    t0 = time.time()
    tif = os.path.join(BASE, "Data", "Clipped_Training", f"{tile_name}.tif")
    out_gpkg = os.path.join(BASE, "Data", "Outputs", f"buildseg_final_{tile_name}.gpkg")
    out_png = os.path.join(BASE, "Data", "Outputs", f"buildseg_final_{tile_name}_overlay.png")

    with rasterio.open(tif) as r:
        img_hwc = r.read([1, 2, 3]).transpose(1, 2, 0)
        transform, crs = r.transform, r.crs
        H, W = r.height, r.width
    print(f"{tile_name}: {W}x{H}", flush=True)

    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    prob = bseg_prob(img_hwc, sess, iname)

    mask = prob > 0.5
    f = img_hwc.astype("float32")
    veg = (2 * f[:, :, 1] - f[:, :, 0] - f[:, :, 2]) > EXG_VEG
    mask &= ~veg

    lab, nblob = ndi.label(mask)
    sizes = ndi.sum_labels(np.ones_like(lab), lab, index=np.arange(1, nblob + 1))
    small = {i + 1 for i, s in enumerate(sizes) if s < MIN_PX}
    if small:
        mask &= ~np.isin(lab, list(small))
        lab, nblob = ndi.label(mask)
    blob_conf = ndi.mean(prob, lab, index=np.arange(1, nblob + 1))
    print(f"mask blobs: {nblob} ({time.time()-t0:.0f}s)", flush=True)

    # raw blobs (parents, for orientation + confidence) — also saved unrefined
    ids, geoms = polygonize(lab, transform)
    raw = gpd.GeoDataFrame({"confidence": [float(blob_conf[i - 1]) for i in ids]},
                           geometry=geoms, crs=crs, index=ids)
    raw_out = raw.copy()
    raw_out["confidence"] = raw_out.confidence.round(3)
    raw_out["area_m2"] = raw_out.geometry.area.round(1)
    raw_gpkg = os.path.join(BASE, "Data", "Outputs", f"buildseg_raw_{tile_name}.gpkg")
    raw_out.to_file(raw_gpkg, driver="GPKG", layer="buildseg_raw")
    print(f"raw -> {raw_gpkg} ({len(raw_out)} blobs, "
          f"{raw_out.area_m2.sum()/1000:.0f}k m2)", flush=True)

    # instance split: watershed on seam energy (cuts follow alleys/eaves),
    # then cascade re-split of big non-rectangular compounds
    seam = seam_energy(img_hwc)
    dist = ndi.distance_transform_edt(mask)
    peaks = peak_local_max(dist, min_distance=MIN_DIST_PX,
                           threshold_abs=SEED_MIN_D, labels=lab)
    markers = np.zeros_like(lab)
    for k, (py, px) in enumerate(peaks, start=1):
        markers[py, px] = k
    px_m2 = abs(transform.a * transform.e)
    pieces_lab = watershed(seam, markers, mask=mask)
    pieces_lab = cascade(pieces_lab, seam, px_m2)
    pids, pgeoms = polygonize(pieces_lab, transform)
    parent_of = ndi.maximum(lab, pieces_lab, index=pids)  # blob id under piece
    pieces = gpd.GeoDataFrame(
        {"parent": [int(p) for p in parent_of],
         "confidence": [float(blob_conf[int(p) - 1]) if p >= 1 else 0.0
                        for p in parent_of]},
        geometry=pgeoms, crs=crs)
    print(f"split pieces: {len(pieces)} ({time.time()-t0:.0f}s)", flush=True)

    # save the blobby split layer (individual roofs, natural outlines)
    split_out = pieces.copy()
    split_out["geometry"] = split_out.geometry.simplify(0.15)
    split_out["confidence"] = split_out.confidence.round(3)
    split_out["area_m2"] = split_out.geometry.area.round(1)
    split_out = split_out[split_out.area_m2 >= rl.MIN_AREA]
    split_gpkg = os.path.join(BASE, "Data", "Outputs", f"buildseg_split_{tile_name}.gpkg")
    split_out.drop(columns=["parent"]).to_file(split_gpkg, driver="GPKG",
                                               layer="buildseg_split")
    print(f"split -> {split_gpkg} ({len(split_out)} pieces, "
          f"{split_out.area_m2.sum()/1000:.0f}k m2)", flush=True)

    final = rl.postprocess(pieces, raw)
    final.to_file(out_gpkg, driver="GPKG", layer="buildseg_final")
    from collections import Counter
    print("methods:", dict(Counter(final.method)))
    print(f"FINAL {tile_name}: {len(final)} buildings, "
          f"{final.area_m2.sum()/1000:.0f}k m2 ({time.time()-t0:.0f}s)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    l, b, rt, t = array_bounds(H, W, transform)
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    ax.imshow(img_hwc, extent=(l, rt, b, t))
    final.boundary.plot(ax=ax, color="yellow", linewidth=0.6)
    ax.set_title(f"{tile_name} — buildseg FINAL v2: {len(final)} buildings, "
                 f"{final.area_m2.sum()/1000:.0f}k m2")
    ax.set_axis_off(); fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    print("overlay ->", out_png, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
