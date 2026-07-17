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
