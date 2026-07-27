"""
Per-polygon confidence score, following the Google Open Buildings convention.

WHY THIS EXISTS
Downstream users (AGT reviewers, the tax roll, the web app) need ONE number per polygon that
says "how sure are we this is a real, correctly-delineated building", plus published cut-offs
so they can filter consistently. Until now the pipeline emitted several raw diagnostics
(mean_prob, cover, frac_in, shift_m, clip_frac) and a class label, leaving every consumer to
invent their own rule.

THE OPEN BUILDINGS CONVENTION WE FOLLOW
  * a single float `confidence` in [0, 1] per building polygon;
  * higher = more likely a genuine building;
  * published threshold bands so users pick a precision/recall trade-off themselves.
    Open Buildings publishes nothing below 0.65 and recommends >= 0.75 for high-confidence
    work, so we mirror those two cut-offs:
        confidence >= 0.75  -> "high"     (use directly)
        0.65 - 0.75         -> "medium"   (usable in bulk, spot-check)
        < 0.65              -> "low"      (route to human review; do not bill on it alone)

HOW OURS IS COMPUTED
Open Buildings' score is a detector output. Ours must also carry the fact that we are
*correcting an existing cadastre*, so a polygon can be a real building yet still be badly
positioned or heavily clipped. The score therefore combines:
    evidence   - how strongly the building-probability mask supports this footprint
    geometry   - is it a sane building shape/size
    position   - how far it had to be moved, and how much was clipped by planarisation
    coverage   - was it truncated at the raster edge

!! CALIBRATION STATUS: UNCALIBRATED (heuristic_v1) !!
Open Buildings' confidence is calibrated against a human-labelled evaluation set, so their
0.8 really means roughly 80% precision. Ours is currently an ORDINAL score: higher is better,
but the number is not yet a probability. It cannot be honestly presented as one until the
exhaustive labelling blocks come back, at which point `calibrate()` below fits the mapping
against human labels and `conf_method` changes to "isotonic_v1". Report it as a ranking score
until then -- that is the difference between a useful triage tool and a misleading metric.
"""
from __future__ import annotations
import numpy as np

CONF_METHOD_DEFAULT = "heuristic_v1_uncalibrated"

# THE DELIVERABLE HAS EXACTLY THREE CLASSES.
# `edge` is an artefact of processing an AOI/tile: it means "less than EDGE_FRAC of this
# polygon fell inside the raster window", which can only happen at a clip boundary. The final
# city-wide product has no clip boundary, so `edge` must not appear in it and must not be
# treated as a quality signal. reclassify_three() resolves any edge polygon on its evidence
# alone, and the edge penalty below is OFF by default for the same reason.
CLASSES = ("keep", "review", "drop")

# Open-Buildings-compatible bands
BAND_HIGH = 0.75
BAND_MED  = 0.65

# evidence blend (mask support under the footprint)
W_MEAN_PROB = 0.60
W_COVER     = 0.40

# evidence thresholds - kept identical to fix_agt_polygons so the 3-class result matches
KEEP_MEAN = KEEP_COVER = 0.40
DROP_MEAN = DROP_COVER = 0.15

# penalty ceilings - each can remove at most this fraction of the score
PEN_SHIFT_MAX = 0.15      # positional uncertainty
PEN_CLIP_MAX  = 0.20      # how much planarisation cut away
PEN_EDGE      = 0.15      # truncation at a clip boundary - OFF by default (see note above)
SHIFT_REF_M   = 10.0      # shift at which the positional penalty saturates


def _col(gdf, name, default=np.nan):
    if name in gdf.columns:
        return gdf[name].to_numpy(dtype="float64", na_value=np.nan)
    return np.full(len(gdf), default, dtype="float64")


_CAL_CACHE = {}

def load_calibration(path: str | None = None):
    """Load the score->precision mapping fitted from human labels, or None if absent.

    This is what turns the score from a ranking into an actual probability, the way Open
    Buildings' published confidence works. Fitted by isotonic (PAV) regression on adjudicated
    candidates; see calibration_v1.json for provenance, n and caveats.
    """
    import json, os
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_v1.json")
    if path in _CAL_CACHE:
        return _CAL_CACHE[path]
    cal = None
    try:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                cal = json.load(f)
            if not cal.get("knots"):
                cal = None
    except Exception:
        cal = None
    _CAL_CACHE[path] = cal
    return cal


def apply_calibration(scores, cal=None):
    """Map raw model scores to calibrated P(building) by monotone interpolation."""
    cal = cal if cal is not None else load_calibration()
    if cal is None:
        return None
    xs = np.array([k["score"] for k in cal["knots"]], dtype="float64")
    ys = np.array([k["precision"] for k in cal["knots"]], dtype="float64")
    s = np.clip(np.asarray(scores, dtype="float64"), 0.0, 1.0)
    return np.clip(np.interp(s, xs, ys, left=ys[0], right=ys[-1]), 0.0, 1.0)


def reclassify_three(gdf, write_col: str = "qa_class", keep_original: bool = True):
    """Collapse the QA classification to the three deliverable classes: keep / review / drop.

    Any `edge` polygon is re-decided on its evidence alone (mean_prob / cover), exactly as if
    it had been fully inside the raster - because in the final product it is. `duplicate` is
    preserved if present, since that is a removal reason rather than a quality band.

    Set keep_original=True to retain the pre-collapse value in `qa_class_raw` for audit.
    """
    import numpy as _np
    if len(gdf) == 0:
        return gdf
    mean_prob = np.clip(np.nan_to_num(_col(gdf, "mean_prob", 0.0), nan=0.0), 0, 1)
    cover     = np.clip(np.nan_to_num(_col(gdf, "cover", 0.0), nan=0.0), 0, 1)

    evidence_class = _np.where((mean_prob >= KEEP_MEAN) | (cover >= KEEP_COVER), "keep",
                     _np.where((mean_prob <  DROP_MEAN) & (cover <  DROP_COVER), "drop",
                               "review"))

    if write_col in gdf.columns:
        current = gdf[write_col].astype(str).to_numpy()
        if keep_original and "qa_class_raw" not in gdf.columns:
            gdf["qa_class_raw"] = current
        # keep duplicate as-is; re-decide edge; leave the other three untouched
        out = _np.where(current == "duplicate", "duplicate",
              _np.where(_np.isin(current, ("edge", "", "nan", "None")), evidence_class, current))
    else:
        out = evidence_class
    gdf[write_col] = out
    return gdf


def confidence_from_fields(gdf, method: str = CONF_METHOD_DEFAULT,
                           apply_edge_penalty: bool = False):
    """Return (confidence float array in [0,1], band string array).

    apply_edge_penalty defaults to False: the deliverable has no `edge` class, so a polygon
    must not be scored down merely because a processing tile clipped it. Pass True only when
    inspecting an intermediate, un-merged tile.

    Expects the pipeline's per-polygon diagnostics; any missing column degrades gracefully
    rather than raising, so this can be applied to older outputs too.
    """
    mean_prob = np.clip(np.nan_to_num(_col(gdf, "mean_prob", 0.0), nan=0.0), 0, 1)
    cover     = np.clip(np.nan_to_num(_col(gdf, "cover", 0.0), nan=0.0), 0, 1)
    frac_in   = np.nan_to_num(_col(gdf, "frac_in", 1.0), nan=1.0)
    shift_m   = np.nan_to_num(_col(gdf, "shift_m", 0.0), nan=0.0)
    clip_frac = np.clip(np.nan_to_num(_col(gdf, "clip_frac", 0.0), nan=0.0), 0, 1)

    evidence = W_MEAN_PROB * mean_prob + W_COVER * cover

    pen_shift = PEN_SHIFT_MAX * np.clip(np.abs(shift_m) / SHIFT_REF_M, 0, 1)
    pen_clip  = PEN_CLIP_MAX * clip_frac
    pen_edge  = (np.where(frac_in < 0.60, PEN_EDGE, 0.0) if apply_edge_penalty
                 else np.zeros_like(frac_in))

    conf = evidence * (1.0 - pen_shift) * (1.0 - pen_clip) * (1.0 - pen_edge)
    conf = np.clip(conf, 0.0, 1.0)

    band = np.where(conf >= BAND_HIGH, "high",
            np.where(conf >= BAND_MED, "medium", "low"))
    return np.round(conf, 3), band


def add_confidence(gdf, method: str = CONF_METHOD_DEFAULT,
                   three_class: bool = True, apply_edge_penalty: bool = False):
    """Attach `confidence`, `conf_band`, `conf_method` (+ collapse to 3 classes by default).

    three_class=True also rewrites `qa_class` to keep/review/drop only, since the deliverable
    has no `edge` class. The pre-collapse value is preserved in `qa_class_raw`.
    """
    if len(gdf) == 0:
        for c in ("confidence", "conf_band", "conf_method"):
            gdf[c] = []
        return gdf
    if three_class:
        gdf = reclassify_three(gdf)
    conf, band = confidence_from_fields(gdf, method, apply_edge_penalty=apply_edge_penalty)
    gdf["conf_heuristic"] = conf          # penalty-adjusted internal triage score

    # If a calibration fitted on human labels is available, publish the CALIBRATED
    # probability as `confidence` - that is what makes it comparable to Open Buildings'
    # score, where 0.8 really means ~80% precision. Otherwise fall back to the heuristic
    # and say so in conf_method, so nobody mistakes a ranking for a probability.
    cal = load_calibration()
    cal_conf = apply_calibration(np.clip(np.nan_to_num(_col(gdf, "mean_prob", 0.0), nan=0.0), 0, 1),
                                 cal) if cal is not None else None
    if cal_conf is not None:
        gdf["confidence"] = np.round(cal_conf, 3)
        gdf["conf_method"] = cal.get("method", "isotonic_pav_v1")
        gdf["conf_band"] = np.where(cal_conf >= BAND_HIGH, "high",
                            np.where(cal_conf >= BAND_MED, "medium", "low"))
    else:
        gdf["confidence"] = conf
        gdf["conf_band"] = band
        gdf["conf_method"] = method
    return gdf


def summarise(gdf):
    """Small dict for the run report: band counts + share, so a run can be judged at a glance."""
    if "confidence" not in gdf.columns or len(gdf) == 0:
        return {}
    conf = gdf["confidence"].to_numpy(dtype="float64", na_value=np.nan)
    n = len(conf)
    out = {"n": int(n),
           "median": float(np.nanmedian(conf)),
           "mean": float(np.nanmean(conf))}
    for name, m in (("high", conf >= BAND_HIGH),
                    ("medium", (conf >= BAND_MED) & (conf < BAND_HIGH)),
                    ("low", conf < BAND_MED)):
        out[name] = int(m.sum())
        out[f"{name}_pct"] = round(100.0 * m.sum() / max(n, 1), 1)
    return out


def calibrate(conf_scores, labels):
    """Fit an isotonic mapping from heuristic score -> empirical precision.

    Call this once the exhaustive labelling blocks return: `labels` is a boolean array
    (True = confirmed building). Persist the fitted model and switch `conf_method` to
    "isotonic_v1" so the published number becomes a genuine probability rather than a rank.
    """
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError as e:
        raise ImportError("calibration needs scikit-learn: pip install scikit-learn") from e
    conf_scores = np.asarray(conf_scores, dtype="float64")
    labels = np.asarray(labels).astype("float64")
    ok = ~np.isnan(conf_scores) & ~np.isnan(labels)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(conf_scores[ok], labels[ok])
    return iso
