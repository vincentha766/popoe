"""Aggregation for BOP AR scoring — the official FLAT (per-instance) calibre.

BOP defines recall as the fraction of *annotated object instances* that are
correct, so every instance carries equal weight and each object contributes in
proportion to its instance count. Averaging per-object recalls with equal
weight per OBJECT (popoe's former behaviour) systematically under-reports:
0.5-0.9 pt on LM-O / YCB-V, while the flat calibre matches the BOP evaluation
server to 0.03 pt.

`metrics/ar.py` and `metrics/vsd.py` report the flat numbers as their primary
output and keep the per-object means as clearly-labelled secondary output, for
reconciling against historical (pre-fix) records.

These helpers are pure numpy so they can be unit-tested and re-run over
persisted error tensors (e.g. `<csv>.vsd_errs.npz`) without bop_toolkit or a
renderer.
"""
from __future__ import annotations

import numpy as np

# BOP19 recall thresholds.
MSSD_THRS = np.arange(0.05, 0.51, 0.05)  # fraction of object diameter
MSPD_THRS = np.arange(5, 51, 5)          # pixels (reference width 640)
VSD_TAUS = np.arange(0.05, 0.51, 0.05)   # tolerance taus (fraction of diameter)
VSD_THS = np.arange(0.05, 0.51, 0.05)    # correctness thresholds on VSD error


def flat_ar_mssd(errs_mm, diams_mm, thrs=MSSD_THRS):
    """Flat AR_MSSD over all annotated instances.

    errs_mm:  (N,) per-instance MSSD errors in mm.
    diams_mm: (N,) diameter of each instance's object in mm (row-aligned).
    """
    errs = np.asarray(errs_mm, dtype=float)
    diams = np.asarray(diams_mm, dtype=float)
    if errs.shape != diams.shape:
        raise ValueError(f"errs {errs.shape} and diams {diams.shape} must align")
    return float(np.mean([(errs < th * diams).mean() for th in thrs]))


def flat_ar_mspd(errs_px, thrs=MSPD_THRS):
    """Flat AR_MSPD over all annotated instances. errs_px: (N,) errors in px."""
    errs = np.asarray(errs_px, dtype=float)
    return float(np.mean([(errs < th).mean() for th in thrs]))


def flat_ar_vsd(errs, ths=VSD_THS):
    """Flat AR_VSD over all annotated instances.

    errs: (N, n_taus) per-instance, per-tau VSD errors (the tensor vsd.py
    persists as `<csv>.vsd_errs.npz`, concatenated across objects). Recall is
    taken over the full tau x threshold grid, then averaged (BOP19).
    """
    arr = np.asarray(errs, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"expected (N, n_taus) errors, got shape {arr.shape}")
    return float(np.mean([[(arr[:, i] < th).mean() for th in ths]
                          for i in range(arr.shape[1])]))


# Reference image width used by BOP for MSPD thresholding (eval_calc_scores.py
# multiplies pixel errors by 640/W before comparing to thr in {5,10,...,50}).
MSPD_REF_WIDTH = 640.0

# BOP official VSD visibility deltas (mm). Source: bop_toolkit
# scripts/eval_calc_errors.py ``vsd_deltas``. ITODD is the outlier at 5 mm.
VSD_DELTA_MM_BY_DATASET = {
    "hb": 15.0,
    "icbin": 15.0,
    "icmi": 15.0,
    "itodd": 5.0,
    "itoddmv": 5.0,
    "lm": 15.0,
    "lmo": 15.0,
    "ruapc": 15.0,
    "tless": 15.0,
    "tudl": 15.0,
    "tyol": 15.0,
    "ycbv": 15.0,
    "hope": 15.0,
}


def mspd_normalize_factor(image_width_px: float, ref_width: float = MSPD_REF_WIDTH) -> float:
    """Factor BOP multiplies raw MSPD (px) by before thresholding.

    err_norm = err_px * (ref_width / image_width). Then compare to thr in
    ``MSPD_THRS`` (5..50). Equivalent: keep err_px and raise thr by W/ref.
    """
    w = float(image_width_px)
    if w <= 0:
        raise ValueError(f"image_width must be positive, got {image_width_px}")
    return ref_width / w


def vsd_delta_mm(dataset: str, default: float = 15.0) -> float:
    """Return the official VSD visibility delta (mm) for a BOP dataset name."""
    key = dataset.strip().lower()
    return float(VSD_DELTA_MM_BY_DATASET.get(key, default))


def assert_csv_covers_targets(csv_keys, targets_path):
    """M1 guard: flat AR denominators by CSV rows alone are wrong if the CSV
    omits targets that appear in ``test_targets_bop19.json`` (those instances
    must count as failures). Raise ``SystemExit`` with a clear count.

    csv_keys: iterable of (scene_id, im_id, obj_id) — any int-like.
    targets_path: path to test_targets_bop19.json (list of dicts).
    """
    import json
    from pathlib import Path

    path = Path(targets_path)
    if not path.is_file():
        return  # no targets file → skip (synthetic / unit tests)
    targets = json.loads(path.read_text())
    tset = {(int(t["scene_id"]), int(t["im_id"]), int(t["obj_id"])) for t in targets}
    cset = {(int(s), int(i), int(o)) for s, i, o in csv_keys}
    missing = tset - cset
    if missing:
        sample = sorted(missing)[:5]
        raise SystemExit(
            f"CSV covers {len(cset)} keys but test_targets has {len(tset)}; "
            f"missing {len(missing)} targets (flat AR would be inflated). "
            f"examples: {sample}"
        )

