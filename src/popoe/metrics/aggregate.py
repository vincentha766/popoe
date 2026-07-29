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
