# Known issues

## NIDS-Net integration + pluggable detection backends (2026-07-16)

Status: DONE (four blocks, each codex-reviewed; 59-test suite green on the
Python-3.12 venv — see below). NIDS-Net added as a third file-based
segmentation source behind a named-backend abstraction; N-way top-M union.

Design decisions worth recording (were not obvious, resolved here not by fiat):

1. **The delivered NIDS files did NOT match the brief's format warning.** The
   task expected fully-stringified fields; the actual
   `data/detections/nids/nids_wa_sappe_{ycbv,lmo}.json` are already
   numerically typed, with **uncompressed** RLE (`counts` a list) — which the
   existing `frPyObjects` branch already decoded byte-correctly (verified vs a
   manual column-major decode). So no adaptation was strictly required for the
   files in hand. The loader still HARDENS for the stringified variant
   (coercion + stringified-RLE parsing) because the documented Box source is
   stringified and a re-download could be; the cost is a few coercions and the
   payoff is that the failure mode is loud, not a silent zero-candidate miss
   (`"1" in [1]` is False). Real files pass through unchanged.

2. **Union filtering is scoped PER SOURCE, not global.** FreeZe's "top-M union
   without filtering" means two sources proposing the same region both survive
   (the scorer disposes). `iou_dedupe` therefore dedupes within a source only.
   For the single-file form this is byte-identical to the old global behaviour
   (all masks share one source), so the evaluated v5 numbers are unaffected.

3. **SAM-6D ISM: LM-O available (pod-generated), YCB-V not.** SAM-6D ISM was
   run on a pod and its LM-O output retrieved
   (`data/detections/sam6d/sam6d_ism_lmo.json`, 20496 dets / 200 imgs); YCB-V
   ISM was never generated, so YCB-V stays a two-way CNOS+NIDS union. The full
   three-way CNOS+SAM-6D+NIDS union is exercised on LM-O
   (`examples/union_smoke.py --dataset lmo --source sam6d=…`: balanced 35/33/33%
   candidate source split over 200 images). No pod opened by this work, no
   inference env installed — only the published/retrieved JSON is consumed.

4. **Union ingestion cross-validated against the gedi merge script.** The
   gedi-era CNOS+SAM-6D LM-O union reference
   (`union_cnos_sam6d_lmo.reference.json`, 33765 = 13269 CNOS + 20496 SAM-6D)
   is the merged detection POOL — raw source-tagged concatenation, no top-M/
   dedup baked in (that runs at segment() time in both stacks). popoe's
   two-source ingestion (`load_bop_detections` per source + combine, which is
   what `BOPDetectionsSegmentor` does into `_by_img`) reproduces it EXACTLY:
   identical multiset of FULL normalised records (all fields, exact scores),
   0 divergences. Pinned by `tests/test_union_reference_xval.py` (skips when
   the local files are absent).

Env note: the full suite needs `open3d`, which has no Python 3.13/3.14 wheel,
so the uv venv is pinned to **3.12** (`.python-version`); `pycocotools` (a real
dep of the RLE decode) is now declared in the `reference` extra. A fresh
default-3.14 venv fails 4 tests on missing open3d/pycocotools — not a code bug.

## Post-fix re-baseline v5: RE-RUN DONE (2026-07-15)

Status: CLOSED — the 07-11 protocol re-run passed on the fixed code (HEAD
4fa47f4). New formal popoe baseline on the 8-object YCB-V subset:

**AR(2/3) = 0.6475** (fresh cache, v5) / **0.6468** (cache-hit, v5b) —
overall agreement 0.07pt; the invalidated pre-fix number was 0.617/0.638.

| obj | # | v5 MSSD | v5 MSPD | v5b MSSD | v5b MSPD |
|-----|-----|--------|--------|---------|---------|
| 5   | 150 | 0.4947 | 0.4793 | 0.4927  | 0.4767  |
| 8   | 75  | 1.0000 | 0.9987 | 1.0000  | 0.9987  |
| 10  | 150 | 0.3907 | 0.2753 | 0.4007  | 0.2847  |
| 14  | 150 | 0.3707 | 0.3587 | 0.4013  | 0.3880  |
| 17  | 75  | 0.8387 | 0.7333 | 0.8467  | 0.7560  |
| 19  | 150 | 0.8387 | 0.7633 | 0.8340  | 0.7573  |
| 20  | 150 | 0.6307 | 0.5587 | 0.6300  | 0.5620  |
| 21  | 75  | 0.8187 | 0.8107 | 0.7627  | 0.7573  |

Protocol identical to v4 (same subset 5,8,10,14,17,19,20,21, same fastSAM_pbr
detections, default env, `--grid 32`, nvdiffrast, AR via freezev2
freezev2_compute_ar_ycbv.py on-pod), fresh cache dir `popoe_cache_ycbv_v5`.
Runtime: 46 min fresh + 28 min cache-hit on one 4090 (pod ycbv-4090-mig9).
CSVs + master log backed up in gedi/ycbv_local_data/ (popoe_ycbv_v5*.csv,
v5_master.log).

Acceptance vs the 07-11 criterion (±3pt/object): all objects within ±3.5pt
except obj21 (−5.6pt MSSD v5→v5b) — the documented knife-edge flip-axis
object (formal itself swings 0.79→0.59), accepted. The previously unstable
obj8 is now saturated (1.000/0.999 in BOTH runs, was 0.20–0.97 pre-fix):
the PCA canonicalisation + w=1 pin + query caching stack holds.

Residuals, both loud (new failure accounting), both negligible:
1. `[FAIL encode_target] obj20` x1 per run: degenerate candidate cloud hits
   `torch.cross` dim mismatch in upstream `gedi.py:188` (`zp.squeeze()`
   collapses a size-1 dim). One CANDIDATE dropped; the target still gets a
   real champion row from other candidates. Upstream-GeDi bug; fix would be
   a guard in feature_extractor.compute.
2. 4 zero-padded rows per run (obj5 scene50 im671/722, obj17 scene51
   im1566/1588), identical in v5/v5b: no usable detection for those images —
   honest misses, not crashes.

## Adversarial review campaign: hidden fallbacks + eval correctness (2026-07-14)

Status: FIXES LANDED, verified (local suite + GPU smoke on A40); re-run
completed 2026-07-15 (see v5 section above) — new baseline 0.6475/0.6468.

Trigger: design review of `CNOSSegmentor._segment_v0` — a silent SAM2→
sliding-window→depth-blob fallback chain hidden inside one segmentor, which
also merge-sorted depth-blob AREA FRACTIONS among DINO COSINES. Generalised
into a platform rule, then the whole repo was swept by four rounds of
external review (codex/gpt-5.5, xhigh), each round fixing what the previous
found, until round 4 returned a single already-fixed finding.

The rule (now in ARCHITECTURE.md + interfaces.BackendUnavailable): a stage
whose backend is missing RAISES; it never substitutes a weaker method under
the same name. Substitution is the caller's policy (segmentor.
FirstAvailableSegmentor), recorded in `chain.last_used` / `Detection.source`.
Runtime failures propagate. Anything that selects a method is config and
belongs in the cache key.

Defects fixed that could have silently biased numbers:

1. **w=1 was never w=1.** `scale_vis`/ChampionScorer are specified against
   w=1 extraction, but `best_encoders` never pinned it, so the env default
   0.5 leaked in: every sweep weight ran at half its label and `s_feat_1`
   re-scored at 0.5. Fixed by pinning `fusion.vis_weight = 1.0` at
   extraction (recipes.py); contract locked by a fusion unit test. THIS
   CHANGES ALL RESULTS — prior CSVs/baselines are not comparable.
2. **NvdiffrastRenderer "depth" was 1/(triangle_id)** (rast channel 3),
   garbage as a depth map; only ever safe as a >0 hit test. Now interpolates
   camera-space z (GPU-verified: median hit depth 0.219 m vs |cam| 0.236 m).
   TrimeshRenderer aligned to the same camera-axis-z convention.
3. **Cache keys under-keyed** (the same class as the 07-11 PCA invariant):
   enc_cfg missed n_views/target_fill/target_canon/vis_weight/skip_vis/
   geom_backbone/dgedi_mode/gedi_path AND the render backend; target keys
   hashed BOP ids, not scene content (rgb/depth/K); `--grid` recorded the
   arg while a pre-set POPOE_TARGET_GRID env silently won. All keyed now —
   existing feature caches are therefore invalid (twice over).
4. **Eval loop swallowed exceptions bare** — real bugs became zero rows
   indistinguishable from "object not found". Now: per-failure print, first
   traceback per (stage, type), end-of-run summary.
5. **inst_count ignored** (latent: LMO/YCB-V are all 1). Now honoured end to
   end. The load-bearing design, forced by review rounds 3-4: completion is
   a WRITER invariant — a finished target emits EXACTLY inst_count rows
   (champions + zero-row padding, missing-image branch included), so resume
   classifies by row count alone. Content-based inference is impossible in
   principle: "completed with fewer champions" and "crashed mid-target" are
   indistinguishable from rows, and real scores format as "0.000000".
   Partial targets' stale rows are dropped by atomic CSV rewrite before
   re-run. Local metrics (ar.py/vsd.py) score one-row-per-target only and
   now HARD-FAIL on multi-instance CSVs instead of silently double-claiming
   GT instances (proper 1-1 assignment: use bop_toolkit, or a future item).

Also: segmentor_cnos imports without torch/cv2 (a chain containing CNOS must
be composable on a box that will route around it); template bank and
Pipeline query cache keyed by (obj_id, mesh_path); BOP ids are only unique
per dataset.

Verification: 30-test local suite green (numpy-only), GPU smoke 18/18 on A40
(chain routing, provenance stamping, metric depth, CNOS end-to-end);
single-instance behaviour proven row-identical through all changes (codex
round-4 clean checks + synthetic resume replay). Re-run criterion for the
re-baselined numbers: fresh-cache and cache-hit runs agree within RANSAC
noise, as per the 07-11 protocol — but expect a NEW baseline, not 0.638:
the w=1 pin changes the operating point of the whole sweep.

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
4. `interfaces.py` (2026-08-23): `Pipeline.run` re-installs the query's PCA
   snapshot before every target encode when the target encoder exposes
   `install_pca` — a THIRD door into the same incoherence, and the one
   `Pipeline` itself left open. `examples/bop_eval.py` already did this, so no
   evaluated number is affected; `Pipeline` did not, so every caller that drives
   more than one object through it was exposed. The failure mode there is not
   run-to-run but call-to-call: `Pipeline` caches query features, so from the
   SECOND call onward `encode_query` never runs and each object's targets are
   projected in whichever object's basis was installed last. Measured on a
   two-object live service (mug + spam, 3 restarts x 2 segmentors): the mug —
   first in the registry, so its basis is overwritten by the spam query — scores
   0.155-0.161 with a 0.4-4.1 deg rotation error on call 1 and 0.080-0.097 with
   a 94-179 deg error on every later call, while the spam can (last, so its own
   basis stays installed) is unaffected. A missing snapshot now raises rather
   than silently using the loaded basis. Regression:
   `tests/test_interfaces.py::test_pipeline_reinstalls_query_pca_on_cached_runs`.

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

## 2026-07-16 · First popoe-native union scoring (supervisor-run, pod L40S)

`bop_eval --sources` (6b3dccf), detections from `data/detections/` mirrored
to the volume. Results (AR(2/3), VSD skipped as usual):

- **YCB-V two-way CNOS+NIDS, v5 protocol subset: 0.6889** vs v5 baseline
  0.6475 (+4.1). No object regresses; obj17 +9.7 / obj21 +10.9 MSSD — the
  knife-edge objects gain most.
- **LM-O three-way CNOS+SAM6D+NIDS, all objects, merge none: 0.7525.**
  First popoe LM-O figure; balanced source usage (35/33/33% per the
  union smoke). Indicatively ~+2 over the script-era two-way mainline
  (cross-stack, ±2.1pt known tolerance applies).

Artefacts: gedi/ycbv_local_data/union_scoring_20260716/ (result+cand CSVs,
logs). Remaining gap: SAM-6D ISM was never generated for YCB-V.

## 2026-07-16 · YCB-V three-way completes: saturation at two sources

SAM-6D ISM detections for YCB-V generated on-pod (56 min on a 4090; 49151
dets / 900 imgs, `sam6d_ism_ycbv.json` — the "never generated" note above is
now stale). Three-way CNOS+SAM6D+NIDS on the v5 subset: **AR(2/3) 0.6899**
vs two-way 0.6889 (+0.1, per-object changes cancel: obj21 +2.1, obj10
−1.3). Reading: NIDS and SAM-6D ISM both propose with SAM-family models —
the third source re-proposes masks the union already holds. Ensemble value
= diversity of proposal/confidence computation, not source count.
Remaining lever: feature-aware mask scoring in the union selector (FreeZe's
mechanism for its larger published gain). Artefacts in
gedi/ycbv_local_data/union_scoring_20260716/.

## 2026-07-16 · A-layer S_coarse arbitration: measured, dataset-asymmetric

Rescan with --score-coarse (both datasets, caches hot) + local rule_replay
(replay baseline matches full runs within 0.04pt). Same-candidate-set AR(2/3):

| rule | YCB-V | LM-O |
|---|---|---|
| baseline fit*s_feat_1*metric | 0.6885 | **0.7755** |
| *s_coarse (plain product) | **0.7137 (+2.5)** | 0.7568 (−1.9) |

YCB-V: gains land exactly on the ICP-attachment objects (obj14 +10.6, obj20
+11.8 MSSD; obj21 gives back 3.2). LM-O (occlusion): pre-ICP feature
consistency is unreliable — s_coarse hurts. Rules remain per-dataset (26-rule
ablation lesson holds). Plain product again optimal where it works.
Cumulative YCB-V subset chain: 0.6475 → 0.6889 (union) → 0.7137 (+S_coarse)
= +6.6 total, matching the magnitude of FreeZe's published ensemble gain with
a measurable two-part decomposition. B-layer (feature-aware RANSAC fitness)
remains the open lever for LM-O.

## 2026-07-17 · Promoted config, full-set, official-metric (VSD included)

Full-set runs of the promoted configs (2fd93cb), VSD via the reference
freezev2_vsd_compute.py:

| | LM-O | YCB-V |
|---|---|---|
| full BOP AR | **0.6896** | **0.8201** |
| MSSD/MSPD/VSD | .7363/.7688/.5638 | .8413/.7879/.8312 |
| script-line formal baseline | 0.6726 | 0.7668 |

YCB-V = 2-way union + use_s_coarse + clamp pooling (all 21 objects, 2-proc
object-split resume). LM-O = 3-way union, baseline rule (s_coarse harmful
there). YCB-V passes SAM-6D's published 0.815 — first time this stack beats
an open-source published method on YCB-V. Ran as a 2-way object-split with
resume-from-copied-CSV; merged by (scene,im,obj) dedupe. Gotcha for the
runbook: `pkill -f bop_eval.py` from an ssh one-liner kills the ssh shell
itself (pattern matches its own cmdline) — kill by PID.

## 2026-07-17 · B-layer closed: feature-aware RANSAC fitness is a negative

GPURansacSolver sweep (best per-dataset detection configs, hot caches):
o3d 0.7137/0.7525 (YCB-V subset / LM-O AR(2/3)) vs gpu 0.6429/0.6419 vs
gpu-feat 0.6498/0.6416. Eq.5-inside-RANSAC is marginal on YCB-V (+0.7) and
flat on LM-O — the relative-ranking hypothesis is refuted; occlusion breaks
hypothesis-level feature ordering too. The dominant factor is the solver
family itself (O3D +7-11pt over the GPU port, matching the historical
grid-16-fast-config gap): the accuracy lever lives in O3D's correspondence
construction/convergence, not the fitness formula. Feature-aware line
closes as: A-layer +2.5 YCB-V (in production) + three measured negatives.

## 2026-07-26 · Solver A/B: table withdrawn, then re-measured on MSSD

ARCHITECTURE.md quoted a median-rotation A/B on YCB-V obj 5 (freeze_ransac 23.4°
/ open3d 1-shot 42.5° / open3d `n_restarts=8` 23.9°) as evidence that "geometry
proposes, features dispose" recovered parity. No raw artefact survived — only
the gedi archive's prose, which gave translation as "~18-20 mm" while popoe's
table printed 17.6/19.5/17.9. That mismatch prompted a rerun; the rerun found
worse than a wrong column.

**Defect 1 — unseeded.** Open3D's RANSAC draws from a global RNG it never seeds,
and 0.17's `registration_ransac_based_on_feature_matching` takes no `seed`
argument, so no run reproduced. Fixed: `Open3DFeatureRansacSolver(seed=...)`
seeds `o3d.utility.random.seed(seed + restart)` per restart. `seed=None` stays
the default so the evaluated o3d mainline is not silently shifted.
`popoe.registration.ransac_pose_estimation` was already seeded
(`default_rng(42)`), which is why only the two open3d rows moved between runs.

**Defect 2 — the median was meaningless.** Seeded, full 150-instance population,
still on raw rotation angle: medians 26.70° / 42.03° / 151.25°. The distribution
is bimodal; all three solvers' non-flipped modes are nearly identical (p25
19-21°) and they differ only in flip rate — 41% / 48% / 51%. Because that
straddles 50%, the median reports which side of the boundary a solver fell on and
swings 125° for 3 points of flip rate. rerank's 151° was not a collapse, it was
50.7% > 50%. Symmetrically, the original 23.9° was a median over 5 draws from a
~50/50 split — close to a coin flip.

**A wrong diagnosis, corrected.** The first write-up of this entry blamed a
symmetry-blind metric on a near-symmetric object, i.e. claimed a 180° flip need
not be an error. That is wrong: `misc.get_symmetry_transformations` returns
**1 transform (identity)** for obj 5 — BOP declares the mustard bottle
NOT symmetric, presumably because its label disambiguates. Those flips are real
failures, worth ~98 mm = 0.5 d under MSSD. The metric still needed replacing,
but for the ordinary reason that a rotation angle is not a pose error, not for
symmetry.

**Re-measured on MSSD** (bop_toolkit `pose_error.mssd`, symmetries from
`models_eval/models_info.json`), seeded, 140/150 instances — the pod died at 140,
the missing 10 are scene 52:

| solver | median MSSD | rec@0.2d | rec@0.5d |
|---|---|---|---|
| freeze_ransac | 42.9 mm (0.218 d) | 0.371 | 0.600 |
| open3d 1-shot | 111.2 mm (0.566 d) | 0.271 | 0.457 |
| open3d n_restarts=8 | 62.7 mm (0.319 d) | 0.343 | 0.521 |

Verdict: the composition **helps** — median MSSD nearly halves vs 1-shot, and
rerank wins head-to-head 72 to 33 — but does **not** reach freeze_ransac, so the
original "parity" claim stays withdrawn. Ordering is stable across thresholds.

`recall@0.1d` is 0.000 for all three. `solver_swap_demo` is not the evaluated
pipeline (FreeZeScorer, GT masks, fixed thresholds) and obj 5 is a known-weak
registration case, so these are relative numbers only.

Lessons: a number with no surviving raw artefact is unverified; a median over a
bimodal near-50/50 distribution hides behind a plausible value; and check the
dataset's own symmetry declaration before invoking symmetry as an explanation.

## 2026-07-27 · Review sweep: the fallback rule was never applied to fusion

Status: FIXES LANDED (`5615c68`, `1df3266`, `4ccbd13`); 254-test suite green on
the Python-3.12 venv. **Nothing here was measured against a run** — these are
read-and-repro findings, so no published number is retracted or restated.

The 07-14 campaign generalised "no hidden fallbacks" into a platform rule and
swept the segmentors and the renderer, where the rule was born. It never
reached `freeze/fusion.py` — which sits upstream of every number the study
reports, not inside one stage's output. Three defects, all of the same shape:

1. **`install_pca(None)` re-fitted the basis on TARGET data.** `pca_vis is
   None` reads like "no snapshot to install", but `fuse()` treats it as "fit
   one here", and a target cloud has plenty of valid points to succeed with. So
   the targets ended up in a basis fitted from target features while the cached
   query features stayed in the query basis — cosines compared across two
   unrelated bases, silently. This is the 07-11 PCA-basis incoherence (obj8: AR
   0.16-0.25 vs 0.79-0.85) arriving through a second door: the sign
   canonicalisation fixed *disagreeing* fits, not a *missing* one.

   Live trigger: `bop_eval` wrote a query entry as two non-atomic files and the
   hit path passed `get_pickle()` through unchecked, so a run killed between the
   writes — or any `rm *.pkl` prune — left arrays with no basis. Now: the guard
   raises, arrays-without-sidecar is not a hit, and the sidecar is written FIRST
   so the `.npz` is the commit marker.

2. **`fuse()` truncated instead of projecting.** With no fittable PCA, or a
   width disagreement, it fell back to `vis_feats[:, :vis_dim]` — the raw first
   64 DINO dims standing in for a PCA projection, under the same name, absent
   from the cache key. Both fallbacks now raise. The one surviving non-PCA path
   substitutes nothing: `n_vis == vis_dim` means there is no reduction to do.

   Order mattered: (1) had to land first. The PCA is always installed on the
   target side now, so this change does not turn small masks into failures — a
   12-point target with 9 NaN geo rows still fuses. Alone it would have.

3. **`feature_aware_score`'s docstring claimed Eq.5 and computed the mean.**
   Documented as `(1/|P_T^sparse|) * Σcos`, implemented as `cos_sims.mean()` —
   the inlier count, a different quantity and a different ranking. The
   codebase already knew: `gpu_ransac` warns about exactly this at its call
   site, having measured the confusion at -31 pt. But the function four solvers
   and ChampionScorer all call still advertised the fixed denominator, so the
   warning sat where a reader does not need it and the misleading formula where
   they do. No behaviour change — mean-over-inliers is the correct half here,
   with the count term arriving separately as `s_icp` — only the docstring, plus
   a test whose two normalisations are 0.75 vs 0.375 so they cannot be confused
   again.

Reachability was checked, not assumed: the evaluated shapes (query 3000 /
target 1024, two-scale GeDi 64-D and FPFH 66-D) all take the projection branch,
and reaching the query-side raise needs >2936 of 3000 points to carry NaN
geometric features.

Lesson: a platform rule is only enforced where someone went and enforced it.
The contract had been stated, tested at the segmentor boundary, and cited in
`ARCHITECTURE.md` for two weeks while the layer feeding every reported number
still had three fallbacks in it.

The same sweep found three smaller things, all since fixed (`d691349`,
`698ffd5`, `05baa0a`):

4. **The seed knob was unreachable.** `Open3DFeatureRansacSolver(seed=...)`
   landed in `42c916e`, but `_build_solver` never passed one and `bop_eval` had
   no flag — so it could only be set by constructing the solver by hand, and
   the mainline `--solver o3d` stayed non-reproducible. `bop_eval --seed` now
   wires it through (default `None` = historical unseeded, nothing shifts). The
   seed stays OUT of the encoder cache key on purpose: it moves poses, not
   features, so keying it would invalidate every pod cache for nothing.
5. **`solver_swap_demo` could not print one of its own quoted columns.** The
   summary emitted recall @0.05/0.1/0.2d while the solver A/B table (now in
   REPRODUCTION.md) quotes @0.2d and @0.5d. Thresholds and labels now come
   from one tuple.
6. **`registration._geometric_prune` was dead** — zero references;
   `ransac_pose_estimation` inlines a simpler two-point check instead.

### A withdrawn finding, and why it was wrong

The sweep also claimed the 07-26 solver A/B numbers had "no surviving raw
artefact" and so shared the status of the table they replaced. **That is
withdrawn.** It came from checking a worktree where `outputs/` — correctly
gitignored — simply did not exist, and was never verified against the machine
that ran the campaign.

The artefacts are intact and properly stamped: `outputs/solver_swap_20260726/`
carries PROVENANCE files with commit hash (`63c5e7d`), pod, fresh-clone path
and timestamp, the run logs, and a README recording the withdrawal. Every cited
number was recomputed from the 140 per-instance rows of `mssd140_run.log` and
reproduces exactly — all three medians, all six recalls, `recall@0.1d = 0.000`
for all three, and the 72/33/35 head-to-head. The run log has no summary block
because the pod died at 140, which is why the figures were derived post-hoc;
that is a recorded circumstance, not a missing artefact.

The table now lives only in REPRODUCTION.md (Solver A/B ledger), which states
outright that it is not a performance claim for popoe. ARCHITECTURE.md
describes the seam and points at the ledger; it does not quote the numbers.

Lesson, second order: "I could not find it" is not "it does not exist". A
provenance complaint is itself a claim about artefacts and needs the same
standard of evidence it demands — check the host that produced the run before
telling someone their numbers are unverified.

### `scale_vis` assumed equal halves (pre-existing) — FIXED

Found by codex round 3 on PR #12, verified **pre-existing** and deliberately
not fixed there — the PR's identity branch is byte-identical to what `main`
already did for this configuration, so it was never a regression. Fixed
separately in PR #13 once the fusion work had landed; the estimate below that
it "touches the evaluated weight-sweep path" turned out to be wrong, see the
resolution at the end.

`recipes.scale_vis` splits fused `[vis|geo]` features with
`vd = feats.shape[1] // 2`, i.e. it assumes the two halves are equal width.
That holds for the evaluated mainline, where `vis_dim` is geo-matched by
construction (64-D GeDi -> 64-D visual -> 128-D fused), and every published
number is unaffected.

It does NOT hold when `POPOE_VIS_DIM` is set to something other than the
geometric width — a real, cache-keyed knob (`enc_cfg` records it). With
`POPOE_VIS_DIM=1536` against 64-D GeDi the fused vector is 1600-D, `vd`
computes to 800, and the selection-time weight sweep scales only the first 800
visual channels while the remaining 736 stay at w=1. Confirmed on `main` and
on the PR: same widths, same mis-scaling, byte-identical output.

So a weight sweep under a non-geo-matched `POPOE_VIS_DIM` silently runs a
different weighting than its label says — the same shape as the 07-14 "w=1 was
never w=1" defect, in the one knob that was not swept then.

**Resolution (PR #13).** The fix was far smaller than the estimate above, for
a reason worth recording: the split does not have to be carried, because it was
already being recorded. `POPOE_VIS_DIM` is part of the encoder cache key
(`enc_cfg['vis_dim']`), so features built under one setting cannot be paired
with another setting's split — a different value is a different key, hence a
different entry. bop_eval therefore already knows the boundary and just had to
pass it.

The second half is a consequence of the PR #12 work: with the truncation and
zero-pad fallbacks gone, the fused visual width **is** `vis_dim` on every
surviving path (PCA projects to `n_components=vis_dim`; the identity path
requires `n_vis == vis_dim`). Before that there was a third case where the
width was whatever a raw slice happened to produce, and no rule would have
held. Removing the fallbacks is what made the split derivable at all.

So `scale_vis(feats, w, vis_dim=None)` splits at `vis_dim`, refusing a boundary
that is not a proper prefix rather than guessing, and `bop_eval` passes the
value from `enc_cfg`. `None` keeps the historical equal-halves answer, so no
existing caller moves. Verified byte-identical on the geo-matched mainline;
under `POPOE_VIS_DIM=1536` all 1536 visual channels are now scaled where 736
were previously left at w=1.

A first attempt had `scale_vis` resolve the split from `POPOE_VIS_DIM` itself
"since that is what the fusion reads". **That was wrong**, and the suite caught
it under a non-default environment: the env var is only the fusion's DEFAULT,
and a caller passing `DinoGeDiFusion(vis_dim=...)` or a pre-fit `pca_vis`
overrides it — the real width is then `n_components`. With a 64-component PCA
supplied and `POPOE_VIS_DIM=1536` exported, the env answer tried to split a
128-D vector at 1536. The split has to come from the caller that knows, not
from a variable that merely influences it.

Fallout worth keeping: `tests/test_fusion.py` had been inheriting the
developer's shell. `POPOE_VIS_DIM=1536` alone flipped four of its tests on
`main` — including `test_output_dims`, which predates all of this work —
because the env changes the width the fused vector actually has. The module now
clears the fusion knobs in an autouse fixture and sets them explicitly where a
test wants one, so the suite passes under a dirty environment too.

Estimating-lesson: "this touches the hot path" was wrong, and wrong in the
direction that defers work. The knob was already keyed, so the information
needed to fix it safely was present the whole time — the cost was in reading
the cache-key construction, not in threading new state.

Review-lesson: the env-fallback error was invisible under the default
environment and only surfaced from deliberately running the suite with the very
knob the change was about. A test suite that never varies the variable under
discussion cannot falsify a claim about it.
