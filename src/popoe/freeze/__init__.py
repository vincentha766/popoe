"""popoe.freeze — the FreeZe-v2 reference method, packaged as pipeline stages.

Everything specific to the FreeZe recipe (DINOv2 + GeDi encoders, the
[vis | geo] fusion rule, the s_coarse*s_fine*s_icp scorer, and the
evaluated-best configuration) lives here; the method-agnostic pipeline
(interfaces, solvers, segmentors, registration primitives, cache, metrics)
stays in the top-level popoe package. A second method would sit beside this
package, not inside it.

This __init__ only re-exports the LIGHT pieces (numpy / scikit-learn).
`popoe.freeze.feature_extractor` imports torch and GeDi and must be imported
explicitly; `popoe.freeze.recipes.best_encoders()` pulls it in lazily.
"""
from popoe.freeze.fusion import DinoGeDiFusion
from popoe.freeze.adapters import (
    FreeZeQueryEncoder, FreeZeTargetEncoder, FreeZeScorer, make_freeze_encoders,
)

__all__ = [
    "DinoGeDiFusion",
    "FreeZeQueryEncoder", "FreeZeTargetEncoder", "FreeZeScorer",
    "make_freeze_encoders",
]
