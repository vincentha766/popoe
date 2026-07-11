# Known issues

## Eval runner does not yet reproduce the formal baseline (2026-07-11)

Status: OPEN — blocks declaring popoe the experiment platform.

8-object YCB-V subset, formal baseline (gedi-repo sweep pipeline) = 0.638
AR(2/3). popoe `examples/bop_eval.py` runs:

| run | ICP iters | AR(2/3) | obj8 (gelatin) | notes |
|-----|-----------|---------|----------------|-------|
| v2  | 50        | 0.610   | 0.971          | after PCA-leak + crash fixes |
| v3  | 2000      | 0.503   | 0.200          | ICP matched to formal settings |
| v3b | 2000      | 0.511   | 0.467          | identical code+cached features as v3 |

Findings so far:

1. **Long ICP destabilises small objects in this runner** (obj8: 0.97 at 50
   iters vs 0.20/0.47 at 2000) even though the formal pipeline uses the same
   2000-iter criteria stably. Suspected interaction with (2).
2. **Large run-to-run variance with identical code and identical cached
   target features** (v3 vs v3b: obj8 differs 27pt). Suspects, in order:
   query resampling nondeterminism (`trimesh.sample_surface_even` unseeded —
   the formal pipeline has the same property but appears far more stable),
   O3D RANSAC nondeterminism, and any remaining metric-vs-canonical space
   mismatch in thresholds.

Diagnosis plan (local, CPU-only — target features are cached): replay
solver/refiner/scorer from the cache for obj8 N times per config
{ICP 50/2000} x {metric/canonical} x {fixed/free query sample}, measure the
variance decomposition. Data: `popoe_cache_ycbv` on the pod volume;
candidate dumps `popoe_ycbv_cands.csv`; result CSVs `popoe_ycbv_subset*.csv`
(backed up in gedi/ycbv_local_data/).
