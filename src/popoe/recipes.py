"""Deprecated shim — the evaluated-best configuration is a FreeZe-method
recipe and moved to popoe.freeze.recipes. Import from there; this path keeps
old imports working."""
from popoe.freeze.recipes import (  # noqa: F401
    SOLVERS, TAU_FRAC, WEIGHTS, YCBV_MERGE_LABELS,
    best_encoders, best_segmentor, scale_vis, stages_for_object,
)
