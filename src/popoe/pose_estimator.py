"""Deprecated shim — this module was split in the freeze/ extraction.

The method-agnostic registration/scoring primitives (feature matching, RANSAC,
ICP, score combination) now live in `popoe.registration`; import from there.
The inline `FreeZeV2` monolith class moved to `examples/freezev2_monolith.py` —
it is a demo / parity oracle for `examples/pipeline_selfcheck.py`, not library
API. The library equivalent is `interfaces.Pipeline` wired via
`popoe.freeze.recipes`.
"""
from popoe.registration import (  # noqa: F401
    top_k_correspondences, feature_aware_score, ransac_pose_estimation,
    icp_refinement, final_score,
)
