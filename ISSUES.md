# Known issues

## Eval runner does not yet reproduce the formal baseline (2026-07-11)

Status: RESOLVED 2026-07-11 (verified) — one residual single-object delta open.

Verification (fresh-cache v4 -> cache-hit v4b, canonical-PCA + query caching):
v4 = 0.6172, v4b = 0.6045; per-object agreement within +-1-4pt (RANSAC noise).
obj8, previously 0.97 -> 0.20 across runs, is now 0.981 -> 0.972. Alignment
with the formal subset baseline (0.638): -2.1pt overall, all objects within
noise or better EXCEPT obj21.

### Residual: obj21 (foam brick) — RESOLVED-AS-EXPLAINED (2026-07-11)
Not a platform defect. Diagnosis chain (all local/offline):
target clouds identical (same masks, same counts, centres within 0.6 mm);
error structures identical (BOTH stacks emit ~180-degree flips at ~2 mm
translation, median raw rot err 178.3 deg on each side). The AR difference is
WHICH flip axis gets selected: sym-aware error median 3.5 deg (formal) vs
91 deg (popoe) — obj21 has one BOP-forgiven 180-degree symmetry, and the
right-vs-wrong-axis variants are a near score TIE under the champion rule
(margins ~1e-3). Formal's specific feature instance happened to discriminate
(right beats wrong 87%, margin +0.022); five popoe query instances all tie
(0.30-0.50). Decisively: formal's OWN two runs swing 0.787 -> 0.589 on this
object — flip-axis selection is a fragile, instance-dependent lottery in the
METHOD, and popoe's draws sit lower in the same distribution.

Real improvement (both stacks, research item, tracked in the gedi study):
appearance-based symmetric-variant arbitration — score the flip variants by
rendered-appearance agreement instead of the near-tied geometric/fused rule.

Platform verdict: popoe eval = ALIGNED (coherence verified v4/v4b; remaining
subset delta -2.1pt is dominated by this one knife-edge object).

**Root cause (proven by local replay + PCA-basis analysis): visual-PCA basis
incoherence between cached target features and re-fitted query features.**
PCA component signs are arbitrary per fit; re-encoding the query in a later
run (different surface sample) re-fits the PCA, and when a TOP component
flips sign, cosine similarity against the cached targets (projected in the
old basis) is scrambled. Measured on obj8: flipped-variance-mass 29-48% <->
AR 0.16-0.25; 3-5% <-> AR 0.79-0.85. This also retro-explains v2 (cache-
build run, self-consistent basis: 0.97) vs v3/v3b (cache-hit runs with
fresh query PCA: 0.20/0.47). The ICP-iteration hypothesis was disproven
(50 vs 2000 iters: no significant effect, fixed-query repeats 0.81-0.90).

Fixes:
1. `fusion.py`: PCA component-sign canonicalisation after fit (largest-
   |loading| entry positive) — any two fits of one object now produce
   compatible bases.
2. `examples/bop_eval.py`: query features + fitted PCA are cached with the
   target features — one basis per object, persisted.
3. `adapters.py`: deterministic query sampling (seed=obj_id).

Verification: fresh-cache run (v4) then cache-hit rerun (v4b) must agree
within RANSAC noise (~±3pt/object) and match the formal subset baseline
(0.638 AR(2/3) over the 8 hard objects).

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
