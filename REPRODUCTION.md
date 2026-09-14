# REPRODUCTION — measured claims

Every headline number cited from popoe belongs in this file, with a commit
hash and an artefact path. Until a row is checked, do not treat a figure as
reproduced.

**Acceptance rule**: same BOP test split, same recipe, full BOP AR within
±0.003 of the comparison number (RANSAC stochasticity; tighten to
bit-identical if the run is seeded).

A popoe-native formal line already exists and needs no parity run: YCB-V
full BOP AR **0.8201**, LM-O **0.6896**, grasp ADD(-S)@0.1d 0.8616 / 0.7816.
The tables below also record a historical script-line re-run through popoe
entrypoints. Do not rewrite one line with the other.

## FreeZeV2 §IV-A experimental-setup conformance audit (2026-08-06)

Source: the Experimental Setup section (**§IV-A**, under "IV. Results") of
[arXiv:2506.09784](https://arxiv.org/abs/2506.09784). That single
tech report is the source for **both** FreeZeV2 (Table II Row 18) and
FreeZeV2-Accurate (Row 19); Row 19 is the BOP Challenge 2024 winner and is
identified by the paper itself as "entry **FreeZeV2.1** in the leaderboard"
(= `method_info/905`). The v2.1-only deltas are audited in a separate table
below, since §IV-A describes the shared setup, not Row 19's extras (those are
stated in §IV-D, Quantitative results).
Code inspected: originally the working tree based on `1d785b7` (dirty at audit
time); the execution-path rows have since been updated against the 2026-08-06
fix-wave commits as each landed. This is a setup/protocol audit, not a result
row or a reproducible run identity. **Full-table re-verification performed at
`a101594` (2026-08-07, adversarial agent, row-by-row against code + paper
text): zero conclusion-level errors, no cross-invalidation from the
patch-style updates; the wording/attribution fixes it listed were applied in
the same pass (triage A3/D8 closed).** **Terminal delta re-check (2026-08-07,
pre-pin): the only behavioural code change since `a101594` is `31d277e`
(--render-score / --mask-m 2n / faithful wrappers re-carry rerank). Its
surface touches exactly one row — `M=N+1`, updated to a declared deliberate
deviation in the frozen recipes — plus the additional-variables paragraph
below; every other row's subject (sampling, aggregation, canvas, DINOv2,
budgets, tau, solvers, timing) is outside that diff. A3 closed at the pin.**

Status meanings: **match** = the paper setting and executed path agree;
**pinned-by-us** = the paper is silent and a frozen local choice fills the gap;
**approximation** = an explicit local substitute; **partial** = the number is
present but its scope or semantics differs; **missing** = the paper protocol is
not implemented by the formal runner.

| Paper setting | Current popoe formal path | Status / disclosure |
|---|---|---|
| CNOS, SAM-6D, NIDS and MUSE, evaluated individually or as an ensemble (§IV-A: "used either individually or as an ensemble"; the ensemble rows 18/19 use all four) | The A and B ensemble recipes below both run the same four official sources (CNOS + SAM6D-441 + NIDS + MUSE); official files for all seven core sets are in `data/detections/`. The single-source arms are the `### faithful-cnos` runbook below (CNOS-FastSAM only). | **Match on composition** for the ensemble arms (four-way, per the rows-18/19 merged segmentation cell). Composition match does not make B paper-faithful — B is tuned elsewhere in the pipeline. |
| 162 templates per object, using the CNOS camera viewpoints | Faithful pins set `POPOE_N_VIEWS=162` and `POPOE_QUERY_VIEWS=ico162`. | **Match.** |
| Poisson disk sampling of the raw query surface points | The query cloud is drawn with `trimesh.sample.sample_surface_even` — rejection-based approximately-even sampling, not a strict Poisson-disk sampler. Measured on the eight LM-O meshes at N=5000: it returns the full count, but its minimum nearest-neighbour spacing is 23% below a blue-noise sampler's (ratio 1.302-1.309) — i.e. it gives up exactly the "minimum inter-point distance" the paper's sentence claims. | **Approximation / pinned-by-us. PRICED 2026-08-10: −0.88 pt** (isolation arm D20, `POPOE_QUERY_SAMPLER=poisson`, LM-O full, faithful-cnos: 0.5568 vs baseline 0.5656; MSSD/MSPD/VSD all move the same way, −1.04/−1.22/−0.37, so it is not a single-leg artifact and it is well outside the 0.19 noise floor). **Swapping in a genuine Poisson-disk sampler makes the faithful arm WORSE**, the same shape as D19: the gap is real as a fidelity record but does not explain the shortfall, and "the faithful arm reads low because of this deviation" is falsified. ⚠️ The paper's [77] is Bridson 2007, a VOLUMETRIC dart-throwing algorithm with no unique meaning on a mesh surface; the arm used Open3D's Yuksel-2015 sample elimination, which satisfies the stated property but IS NOT the cited algorithm. Do not write this up as "implementing [77]". |
| Per-point multi-view feature aggregation ("weighted average") | Aggregation is a binary-visibility equal-weight mean (`sum / visible_count`): a view that sees the point weighs 1, others 0. The paper does not publish its weight formula. | **Pinned-by-us.** Do not claim the paper confirms the absence of e.g. view-angle weights. |
| Retain raw query points visible in at least `V=18` views | Faithful pins set `POPOE_QUERY_MIN_VIEWS=18`; filtering is implemented in `src/popoe/freeze/feature_extractor.py`. | **Match.** |
| Render at 480×480 with the object occupying approximately 50% of width/height | Faithful pins use `POPOE_QUERY_CANON=476`, `POPOE_QUERY_FILL=0.5` and `POPOE_QUERY_FILL_MODE=effective`. 476 is the local DINO patch-grid-compatible substitute. The fill knob was historically INERT (mesh pre-scale and camera radius cancelled, leaving a constant ~0.58 frame fraction for the 60° camera — audit P2); `effective` mode sets the camera radius so the largest side actually spans `fill` of the canvas. Legacy mode remains the tuned/historical identity. | **Approximation.** Always disclose 476/0.5-effective, not “exact 480×480”. |
| ViT-giant DINOv2 patch features from intermediate layers, following FoundPose | The backbone is `dinov2_vitg14_reg`. The implementation selects one inferred FoundPose-style layer (block 30 for ViT-g), because the public text does not pin an exact block/list. | **Partial / pinned-by-us.** Backbone matches; layer selection is a local inference. |
| Query point set 5k | The adapter samples 5k surface points before the `V=18` visibility gate; the final retained query set can therefore contain fewer than 5k points. | **Partial.** The paper's stated 5k budget is not enforced after filtering. |
| Dense target point set 3k | Faithful recipes pin `POPOE_TARGET_DENSE=3000` alongside `--icp-dense --icp-dense-max 3000`: the GeDi neighbourhood and the ICP cloud subsample through the same fixed-seed rule (`adapters.fixed_seed_subsample`) over the same index space, reaching the identical 3k cloud. | **Match** (one shared `P_T^dense`, as the paper describes). |
| Sparse target point set at most 256 | Faithful recipes pin `POPOE_TARGET_PAPER_GRID=1`: minimal axis-aligned SQUARE bbox, sparse targets = its 16x16 patch CENTRES, per-patch direct feature assignment (no bilinear), off-object centres dropped with no fallback. Legacy rectangular-grid/bilinear remains the tuned identity. | **Match on the faithful path.** |
| Top-`k=10` feature correspondences | Faithful recipes run `--solver gpu-feat`, whose `GPURansacSolver` samples from top-`k=10` feature correspondences natively (`k=10` is the solver default, pinned in code). `--corr-topk` is an o3d-only knob and is NOT passed — `_build_solver` refuses it on non-o3d solvers. | **Match.** |
| Localization keeps `M=N+1` proposals | `examples/bop_eval.py` floors the per-(source, label) mask budget PER TARGET (`floored_topk`, passed at each `segment()` call). The CLI default `--mask-m n1` is the paper's `N+1`; the FROZEN recipes pass `--mask-m 2n` (= `max(--topk, 2·inst_count)`) per decision 13's v2.1 alignment — at N=1 the two floors coincide (both 2), so the executed path deviates from §IV-A's `N+1` only on multi-instance targets. | **Deliberate deviation in the frozen recipes** (v2.1 `M=2N`, decision 13; see the v2.1-only deltas table). The base `N+1` path remains the default and is byte-identical on single-instance targets. |
| Detection uses `M=100` and discards masks below `tau_mask=0.4` | The formal runner consumes BOP `test_targets_bop19.json` and exposes neither this detection-mode proposal count nor the paper confidence cutoff. | **Missing.** Current formal runs are localization-protocol runs. |
| Two-scale GeDi: 32D per scale, neighborhoods at 30% and 40% of object diameter | `_TwoScaleGeDi` concatenates two 32D descriptors; faithful diameter normalization makes radii 0.3 and 0.4 object diameter. | **Match.** |
| PCA/fusion output dimension 128 | Two-scale geometry is 64D; visual PCA defaults to 64D; normalized concatenation produces 128D. | **Match.** |
| RANSAC inlier and ICP thresholds are 3% of object diameter | Faithful recipes pass `--tau-diameter`; the threshold is 3% of the BOP `models_info` diameter, loaded once in `bop_eval` and fed as ONE basis to RANSAC tau_inlier, ICP tau_ICP and the feature-score radius. (The canonical frame's diameter normalisation is a separate path serving the GeDi radii, not the source of tau.) | **Match.** |
| Parallel GPU RANSAC, 10,000 iterations, selected with the feature-aware score | Faithful recipes run `--solver gpu-feat`: `GPURansacSolver` with `fitness="feature"`, the fixed-denominator Eq. 5 hypothesis selection, at 10,000 iterations (two further local pins the paper does not state: `min_inliers=6` and the 0.9 relative-edge-length triplet check). Tuned recipes keep `--solver o3d` (CPU Open3D geometry RANSAC) as a declared tuning choice — it measured +6-8 pt over the gpu path historically and is NOT the paper's selector. | **Match on the hypothesis selector; incomplete on triplet pruning**. See the row below. |
| Query-view rendering appearance: the paper specifies WHAT is rendered (multi-view renders of the CAD model, DINOv2 patch features) but **not how the pixels are produced** | Five values pinned by us, none stated in the paper, all in `freeze/feature_extractor.py`: **untextured albedo `(180,160,140)/255`** (L473), **background `(200,200,200)/255`** (L475), **Lambertian floor `0.1`** with a headlight source at the camera (L446-448), **occlusion tolerance `0.05 × max extent`** (L757), **FOV `60°`** (L613, L414). | **Pinned by us — five unregistered degrees of freedom, logged 2026-08-10.** ⚠️ **These determine every pixel of the DINOv2 query half.** The table previously logged canvas size and fill ratio but not the values that decide the pixel content itself. Not priced: isolating them needs a re-encode (the cache key covers them via `enc_cfg`, so a change would invalidate the cache rather than silently reuse it — that part is sound). Registered so that "how do you rule out the gap coming from rendering conventions?" has an entry to point at, rather than a gap in the ledger. |
| Triplet pruning before transformation estimation: *"reject triplets in which the distances between matched points exceed a geometric threshold **or** relative edge lengths … are inconsistent"* (v1 paper Sec. III) | The `or` joins **two independent** rejection conditions. `gpu_ransac.py` L90-95 implements **only** the relative-edge-length ratio (`_EDGE_RATIO = 0.9`, a value the paper does not state, borrowed from the Open3D default). The **geometric-distance** condition is **not implemented** on the gpu path; the distance test only happens after Kabsch, on the full correspondence pool (L90-117), which is inlier counting, not sample pruning. Open3D's path does have it as `CorrespondenceCheckerBasedOnDistance(tau_inlier)`. | **Deviation — implementation incomplete relative to the paper.** This is distinct from the deliberate `o3d` tuning choice above: the paper states the step, and the faithful path does not perform it. **Priced 2026-08-09**: the isolation arm (`--solver gpu-feat-dist`, LM-O full, everything else byte-identical to the faithful recipe) scores 0.5616 against the 0.5656 baseline — **−0.40 pt, i.e. adding the paper's missing condition makes the faithful arm slightly WORSE**, consistently across all three legs. Above the 0.19 pt noise floor, so not noise, but small. The gap is therefore a genuine fidelity deviation that explains **none** of the +5.36 pt solver-substitution recovery, and the natural guess that the faithful arm is depressed by this omission is **refuted**. |
| Timing hardware: NVIDIA A40 and Xeon Silver 4316 @ 2.30 GHz | Recorded project runs use the lab 4×RTX 4090 host or other stated infrastructure; recipes do not assert the paper CPU/GPU model. | **Hardware mismatch.** Accuracy results may still be compared with full disclosure; runtime/FPS must not be presented as paper-hardware parity. |

Additional algorithmic variables: BOTH arms enable `--render-rerank
--render-score --mask-m 2n` (decision 13, 2026-08-07 — the three v2.1
components, same-knob across arms; none is specified in §IV-A, and all three
are pinned-by-us approximations reported as such — see the v2.1-only deltas
table below). Decision 9's revision (faithful carries no rerank, triage D5)
is superseded by decision 13.

### v2.1-only deltas (Table II Row 19 = FreeZeV2-Accurate = `method_info/905`)

Row 19 is the strongest configuration **in Table II of this report**, and has
full public per-set scores on the BOP leaderboard (LM-O 0.771 / YCB-V 0.915;
§IV-D reports mean 82.1 AR; method 905 matches Row 19 digit-for-digit across
all seven sets).

**It is not the strongest published FreeZe config, and not the leaderboard
top.** `method_info/1063` = **FreeZeV2.2** (2025-05-31) scores ARCore **0.833**
vs v2.1's 0.821 — LM-O 0.777 / YCB-V 0.918. Its stated delta is
feature-similarity in the RANSAC fitness, and its **segmentor composition is
publicly undeclared**, so v2.2 is a frontier reference only, never a
like-for-like target. Above v2.2 the board has
WAPR.v2 at 0.845 and FRTPose-WAPR.v2 at 0.844 (its Default variant at 0.837). So qualify the superlative every time: Row 19 is the
best config *in this paper*, v2.1 is the best *documented and decomposable
ceiling* for this audit, and neither is state of the art.

Row 19 is nonetheless the right **ceiling** reference for this audit, because it
is the strongest config whose recipe is documented well enough to enumerate
deltas at all. Each delta has a different kind of unavailability, and
they must not be collapsed into one "missing":

| v2.1 delta | Status | Reason / what it would take |
|---|---|---|
| `M = 2N` masks per segmentation model | **Implemented (2026-08-07, `31d277e`): `--mask-m 2n`**, per-target floor `max(--topk, 2·inst_count)`; at N=1 identical to N+1, so single-instance targets are byte-same across modes. In BOTH arms' frozen recipes per decision 13. | **Two distinct sources, do not conflate.** (a) *Where `2N` is stated*: §IV-D prose on Row 19 only — "increases the number of processed masks up to `M = 2N`". No ablation, no per-set numbers, no timing for `2N` anywhere in the report. (b) *What Table V actually ablates*: `M ∈ {N, N+1, N+2}` for the **base FreeZeV2** localization protocol (73.7 / 75.4 / 75.6 mean AR at 1.2 / 1.5 / 1.7 s), establishing that `N+1` is the paper's default and that returns are already flattening by `N+2` (+0.2). Table V therefore **does not validate `2N`** — it neither measures it nor bounds it; the `N+2` trend is only weak evidence that `2N`'s gain is small and its cost is not. What makes `2N` alignable is that the *quantity* `M` is public and parameter-free, not that it was ablated. The per-target `N+1` prerequisite is in place (`floored_topk`), and the coefficient change landed as `--mask-m 2n`; report any `2N` run as our own measurement, since the paper gives no `2N` number to compare against. |
| Symmetry-Aware Refinement (SAR) | **Approximation ceiling: implementable, not verifiable as equivalent.** | §IV-D only says v2.1 "integrates Symmetry-Aware Refinement (SAR) [9]" — ref [9] is FreeZe v1 (`2312.00947v3` §3.6, "based on rendering and visual features"). So the spec lives in a *different* paper and there is no public code. popoe's `--render-rerank` reorders a fixed PCA-axis variant set only (champion + three 180-degree flips + az90/az270; see the tuned recipes' Cautions rows). That is our own symmetry enumeration, not a port of v1 SAR. Any implementation must be disclosed as an approximation of SAR, never as SAR. Decision 13 (2026-08-07) puts `--render-rerank` in BOTH arms' frozen recipes (previously tuned-only after the decision-9 revision). |
| Improved scoring by comparing visual features of input image vs rendered pose | **Implemented as a pinned-by-us approximation (2026-08-07, `31d277e`): `--render-score`** — champion selection multiplies in the clamped `sar_ti` (input-vs-render DINOv2 patch cosine) the rerank stage leaves on every candidate; zero extra renders. In BOTH arms' frozen recipes per decision 13. | §IV-D gives one prose sentence, no equation and no parameters — the factor form (unit-exponent multiplicative, clamped at 0, the `use_s_coarse` arbitration shape) is our own choice and must be disclosed as such, never as the official component. Not reconstructible to a verifiable spec from the public text. |

**Four-source masks including MUSE are NOT a v2.1-only delta** — Rows 18 and 19
use the same four sources, so MUSE belongs to the shared §IV-A setup row above,
not here. Row 19's deltas over Row 18 are exactly the three in this table.

MUSE mask availability — **RESOLVED 2026-08-06 against `method_info/873`.**

MUSE has no public code, but **official masks are downloadable for all seven
BOP-Classic-Core sets.** The two files already in `data/detections/muse/`
(downloaded 2026-07-26, SHA256s in
`outputs/seg_ap_20260725T223014Z/OFFICIAL_JSON_ACQUISITION.md`) are members of
the same authored batch as the other five:

| Dataset | `sub_info` | Batch (2025-08-26) | Records | SHA256 |
|---|---|---|---:|---|
| LM-O | **29108** | 05:14 | 7146 | `55061983089d6236c19cb9b6a8a6c754388d146287be45ec40ceb9c32dbe3003` |
| YCB-V | **29113** | 05:16 | 16902 | `b4703a218d13f707d47556b2733eeddc38fea7d89bf927d113da25349c74f497` |
| IC-BIN | 29109 | 05:14 | 4731 | `34a2a40b3c716bb3c36b0739d49ebc885019cb6d97079d4e5e1ba9c743ed1427` |
| TUD-L | 29110 | 05:15 | 15736 | `38dc40cfa75f22a74f1f85cb10fb2283adb99db65ea496dff68bf216beeccb8b` |
| T-LESS | 29111 | 05:15 | 26511 | `78bcdab72d0eac44ab5b8477eec9e229fdaa2e61fdc69bcec48be46a3f230482` |
| ITODD | 29112 | 05:15 | 6320 | `2d34ebce3a464f129f6cdc8770686df56869eafa8bd8fff135fbfecb3c65813a` |
| HB | **29063** | 05:15 | 6440 | `c0e0802a3db1e2394507099098ed5000208d93e1701f8b19850d6cd6d7d59d1d` |

All seven are now downloaded (2026-08-06). MUSE was therefore not an
availability limit when the formal recipe froze on 2026-07-30, and
`muse-repro` is *not* required to cover them. Earlier claims that masks
were unavailable, or published for only two sets, were wrong.

**Vintage caveat vs FreeZeV2.1(905) — added 2026-08-24.** The files above are
the 2025-08-26 batch, and that batch is MUSE's *earliest* BOP submission.
FreeZeV2.1's own leaderboard submissions are dated **2024-11-29/30**
(`method_info/905`) — nine months earlier. Whatever MUSE masks 905 consumed,
they are therefore **not** the artefacts we use. Composition still matches
(both sides are CNOS + SAM-6D + NIDS + MUSE combined as an unfiltered union),
but **"the same four sources" must never be read as "the same four files"**.
The same distinction applies to SAM-6D, where BOP carries several variants
spanning 5.3 pt of mean segmentation AP and the paper names only "SAM-6D": we
pin 441, they do not say.

Direction and size of the resulting bias:

- **Direction is unknown but plausibly against us.** If MUSE improved over those
  nine months, our detection input is the stronger one, which would make the
  pose-stage residual we report a lower bound rather than an upper one.
- **Size is bounded from outside by the leaderboard itself.** FreeZeV2 (756,
  three sources, submitted 2024-09) to FreeZeV2.1 (905, four sources *plus* SAR
  and the other v2.1 deltas) is +0.7 pt on LM-O and +0.9 pt on YCB-V **in
  total**, so MUSE's marginal contribution on their side is at most that —
  small next to the residual either comparator row carries.
- **Our own leave-MUSE-out margin has never been measured.** A four-way minus
  MUSE arm would bound this from our side instead of theirs; until it exists,
  this caveat rests on their numbers, not ours.

Two traps if these files are re-fetched:

1. **HB is `29063`**, outside the otherwise contiguous 29104–29124 block. Do not
   infer IDs by counting.
2. **The 05:47–05:48 submissions (`29115`–`29121`) are a different TASK, not a
   re-run** — "Model-based 2D detection of unseen objects", bbox only, with **no
   `segmentation` field at all** (verified 2026-08-06). Unusable as mask input;
   their AP is box AP and is not comparable to the seg AP above. Take masks only
   from the 05:14–05:16 segmentation batch. Full seg-vs-det AP table and the
   evaluator-spread caveat on LM-O (public 0.477 vs local PyPI 0.471 vs BOP-fork
   0.483) are in `data/detections/muse/PROVENANCE.md` — that file is a tracked
   `.gitignore` exception precisely so this provenance survives a fresh clone.

`muse-repro` (`src/popoe/segmentor_muse.py`, 1149-line from-paper
reimplementation; seg-AP YCB-V 0.684 vs official 0.690, LM-O 0.388 vs 0.471)
therefore keeps its original role — evidence that the method can be
reimplemented from the paper, not a replacement for official masks in pose
runs.

Consequence for the dissertation: all three components now run in both arms
(decision 13: rerank≈SAR, `--mask-m 2n`, `--render-score`), but exact Row 19
recipe parity remains unverifiable because SAR and render scoring are
underspecified — our versions are pinned-by-us approximations, not ports. The
distance to 905 stays confounded by those three implementation deviations —
label them whenever a triple is written against it. The detection-matched residual lives at A/four-way vs
FreeZeV2.1(905): matched in **detection composition** (the same four sources),
but `--render-rerank` (a popoe-scoped variant of the official SAR: it
reorders a fixed PCA-axis variant set only) means the *pose stage* is still not fully
recipe-matched in either direction. Prose must say "detection-matched",
never "recipe-matched". Paper Row 18 (four-way, no SAR, self-reported only —
no leaderboard counterpart) is citable as an auxiliary reference, not as the
formal comparator.

Required follow-up before claiming exact setup parity:

- [x] Route the paper-faithful recipe through 10,000-iteration GPU
  feature-aware RANSAC and verify that Eq. 5 is the executed selector
  (done — faithful recipes run `--solver gpu-feat`: `GPURansacSolver`
  defaults `iters=10000`, ranks hypotheses by the fixed-denominator Eq. 5
  `fitness="feature"`, and matches top-`k=10` query NNs per TARGET point —
  the paper's Eq. 3 direction — with the mutual filter off).
- [x] Define one 3k dense target cloud and reuse it consistently for GeDi and
  ICP (done — `POPOE_TARGET_DENSE=3000` + `--icp-dense-max 3000` draw through
  one `fixed_seed_subsample` over the same index space; a mismatch between
  the two values refuses to start).
- [x] Implement per-target `M=N+1` (done — `floored_topk` + per-call
  `segment(topk=...)`; unblocks the v2.1 `M=2N` row above as a coefficient
  change). Still open: the separate detection protocol with `M=100`,
  `tau_mask=0.4` (6D detection task only, not needed for localization runs).
- [ ] Resolve or explicitly preserve the 480→476, post-visibility 5k, and
  sparse-target sampling approximations.
- [ ] Add a configuration/contract test covering the complete paper setup.

## Two-line formal recipes

> Formal score = BOP evaluation server only. Local full AR in
> `AR_SUMMARY.md` is a development self-check, not the dissertation score.
> Code identity is **FROZEN**: tag `eighteen-run-freeze-20260807`; each
> script refuses to run unless `POPOE_PIN` is the tag's dereferenced full
> sha. `RUN_ROOT` is required (no default). Do not quote numbers from the
> retired tag `twoline-rerank-fix-20260731` @ `509072e`, or from the
> 2026-07-30 batch (`twoline-prep-20260730a`): that batch re-ICP'd only
> flipped variants at a 4–10× too-loose threshold. `bop_eval` resumes by
> row count, so a stale `poses.csv` / `cand.csv` must be moved aside
> before a re-run or every target is declared done with the bad data.
>
> Values marked **pinned-by-us** are frozen project choices where the
> public recipe is silent: `--seed 42`, A-line Eq.7 unit exponents
> (`alpha=beta=gamma=1`) for `--use-s-coarse`, `POPOE_QUERY_CANON=476`
> (paper names 480²/50%), and `--trans-nms 0.05` (Sec. III-F names
> translation NMS but no radius). `--render-rerank` is score-affecting,
> so those runs need fresh pose/candidate CSVs.
>
> **Smoke first**: same block with `--objs 1` and `smoke_`-prefixed
> `--out/--cand-csv`. A smoke must assert a number, not just survival.
> **S1** `python scripts/check_rerank_symmetry.py smoke_cand.csv` — fail
> if flipped/unflipped `s_icp` median ratio **> 1.4**. **S2** `ar_flat`
> on the smoke poses: reference `tuned-4way` LM-O `--objs 1` needs
> **AR(2/3) ≥ 0.65**. **S3** (env pins, rerank lines, no Traceback, rows
> > 0) is necessary, never sufficient.
>
> **Post-run (no GPU)**: per dataset dir fill the remaining four artifacts —
> `RECIPE.md` (copy the exact block + commit + date), `AR_SUMMARY.md`
> (local `ar_flat`/VSD scripts; self-check only), `bop_server.md` (score +
> submission id after the private upload), `grasp_summary.md` (same-CSV
> ADD(-S) via `python -m popoe.metrics.grasp`).

The layout is **flat** — raw and derived artifacts live
side by side in one directory and are distinguished by **filename**, not by
`raw/` / `derived/` subdirectories:

```
$RUN_ROOT/{recipe}/{dataset}/
  poses.csv        # RAW · main pose CSV (when sharded: the MERGED result)
  cand.csv         # RAW · candidate-level dump (--cand-csv; replay/ablation input)
  RECIPE.md        # RAW · flags, detection sources, commit, seed, date
  *.log            # RAW · AR_FULL / AR_LOCAL / vsd run + self-check logs
  submission.csv   # DERIVED · time-normalized CSV actually uploaded to BOP
  AR_SUMMARY.md    # DERIVED · local full AR + MSSD/MSPD/VSD self-check only
  bop_server.md    # DERIVED · official BOP server score + submission id
  grasp_summary.md # DERIVED · ADD(-S); Phase D LM-O/YCB-V only (not Phase E)
  replays/         # DERIVED · Ch5 replay outputs, when this run carries any
```

Sharding: shard directories are **siblings** named `{dataset}_s{A,B,C}`; the merged
result is written to the un-suffixed `{dataset}/`. Shard directories need only
`poses.csv` / `cand.csv` / `RECIPE.md` / logs.

Because there is no directory-level raw/derived split, the "raw is never
overwritten" guarantee rests on filenames: **never rewrite `poses.csv` or
`cand.csv` in place.** Time normalization writes `submission.csv`; merge writes the
un-suffixed sibling directory; replays write `replays/`.

> Verified against the GPU host on 2026-08-09
> (`results/run18_20260807/{recipe}/{dataset}/`). Earlier notes described two
> different `raw/`+`derived/` layouts; neither matched the frozen commands, and
> both have been corrected to the flat layout above rather than changing any
> output path (changing output paths would change run identity).

### faithful-cnos

This block is the CNOS-only runner. The former `scripts/faithful_eval.sh` /
`scripts/faithful_run.sh` wrappers duplicated it (mismatched output names,
no `POPOE_PIN`) and were removed.

| Field | Frozen value |
|---|---|
| Report point | A / single-source |
| Code | **FROZEN** — tag `eighteen-run-freeze-20260807`; the operator supplies `POPOE_PIN=<its dereferenced full sha>` at run time (script refuses without it) and RUN_SPEC requires HEAD == PIN == tag^{commit}; the retired pin `twoline-rerank-fix-20260731` @ `509072e` predates the 2026-08-06 fix wave and identifies the voided runs only |
| Datasets | LM-O + YCB-V; one full BOP test run each |
| Detection inputs | CNOS only: `data/detections/cnos/cnos-fastsam_lmo-test.json`, `data/detections/cnos/cnos-fastsam_ycbv-test.json` |
| Scoring | Paper Eq.7 three-term form with `--use-s-coarse` and `--eq5-terms` (both feature terms in the Eq.5 formulation: target->query top-k pool, fixed \|P_T^sparse\| denominator); Eq.7 exponents are **pinned-by-us** to unit exponents |
| Leaderboard comparator | A / single-source -> FreeZe(CNOS) LM-O/YCB-V = 0.689 / 0.853 |
| Artifacts | `$RUN/{lmo,ycbv}/` each contains `poses.csv`, `cand.csv`, `RECIPE.md`, `AR_SUMMARY.md`, `bop_server.md`, `grasp_summary.md` |
| Cautions | **Rerank + render score + M=2N carried (decision 13)**: the faithful arms align to v2.1 (Row 19), which carries SAR, M=2N and render scoring — popoe's `--render-rerank --render-score --mask-m 2n` are pinned-by-us approximations of those three (see the v2.1-only deltas table), disclosed as such, never as the official components. Dense resampling uses `rng(0)` where invoked and is independent of `--seed`. Encoding degradation is explicit in logs as `DEGRADE`. These runs are not bit/row comparable to historical anchors because seed and implementation fixes are new variables; the pre-decision-13 dev anchors (faithful 0.7314 / tuned 0.8174) are void — superseded by the 2026-08-07 re-anchor @ `efe97ab`: faithful-4way v4 **0.7369** / tuned-4way r3 **0.8243** AR(2/3) (LM-O obj-1 smoke, official-441 four-way). |

```bash
set -euo pipefail

POPOE="${POPOE:-/workspace/popoe}"
BOP="${BOP:-/workspace/bop_data}"
DET="${DET:-$POPOE/data/detections}"
: "${RUN_ROOT:?set RUN_ROOT=<fresh run root, e.g. /workspace/results/run18_20260807>}"
RUN="$RUN_ROOT/faithful-cnos"
PY="${PY:-python}"
SEED=42

cd "$POPOE"
# Code identity: tag eighteen-run-freeze-20260807. The operator supplies its
# dereferenced full sha explicitly — never a bare tag name (stale-tag guard,
# RUN_SPEC). The old pin (twoline-rerank-fix-20260731 @ 509072e) predates the
# 2026-08-06 fix wave and identifies the voided runs only.
: "${POPOE_PIN:?set POPOE_PIN=<dereferenced full sha of tag eighteen-run-freeze-20260807>}"
if [ "$(git rev-parse HEAD)" != "$POPOE_PIN" ]; then
  echo "wrong popoe checkout; need POPOE_PIN=$POPOE_PIN (got HEAD=$(git rev-parse HEAD))" >&2
  exit 1
fi
# Rerank file guard — faithful re-carries rerank per decision 13 (2026-08-07):
for req in scripts/check_rerank_symmetry.py scripts/sar_render_compare.py; do
if [ ! -f "$req" ]; then
  echo "missing $req — stale tag tip (render_rerank loads sar_render_compare at RUNTIME); fetch --tags --force" >&2
  exit 1
fi
done
if [ -n "$(git status --porcelain)" ]; then
  echo "refusing dirty popoe worktree" >&2
  exit 1
fi

export BOP DET RUN SEED
export OMP_NUM_THREADS=8
export TORCH_HOME="${TORCH_HOME:-/workspace/torch_cache}"
export POPOE_GEDI_PATH=/workspace/gedi
export POPOE_BOP_TOOLKIT=/workspace/bop_toolkit
unset POPOE_TARGET_GRID POPOE_TARGET_CANON POPOE_TARGET_FILL POPOE_TARGET_CROP
unset POPOE_VIS_DIM POPOE_VIS_WEIGHT POPOE_SKIP_VIS POPOE_DINO_LAYER
unset POPOE_TWO_SCALE_GEDI POPOE_DGEDI_MODE POPOE_GEOM_BACKBONE POPOE_MESH_SHADING
unset POPOE_FPFH_RADII POPOE_FPFH_VOXEL_FRAC POPOE_FPFH_NORMAL_FRAC POPOE_FPFH_ORIENT
export POPOE_CANON_BASIS=diameter
export POPOE_QUERY_POINTS=5000
export POPOE_N_VIEWS=162
export POPOE_QUERY_CANON=476
export POPOE_QUERY_FILL=0.5
export POPOE_QUERY_FILL_MODE=effective
export POPOE_QUERY_MIN_VIEWS=18
export POPOE_QUERY_VIEWS=ico162
export POPOE_TARGET_DENSE=3000
export POPOE_TARGET_PAPER_GRID=1

mkdir -p "$RUN/lmo" "$RUN/ycbv"
for f in "$RUN/lmo/poses.csv" "$RUN/lmo/cand.csv" \
         "$RUN/ycbv/poses.csv" "$RUN/ycbv/cand.csv"; do
  # NOT `test ! -e a && test ! -e b`: under set -e a failing left-hand test
  # is errexit-exempt and the guard silently passes (second-review P0).
  if [ -e "$f" ]; then echo "refusing: $f exists — resume keeps old rows; use a FRESH --out" >&2; exit 1; fi
done

"$PY" examples/bop_eval.py \
  --bop "$BOP/lmo" --dataset lmo \
  --detections "$DET/cnos/cnos-fastsam_lmo-test.json" \
  --merge none --topk 2 --grid 16 --solver gpu-feat --seed "$SEED" \
  --weights 1.0 \
  --use-s-coarse \
  --eq5-terms \
  --min-mask-pixels 0 --mask-iou-dedupe 1.1 \
  --tau-diameter \
  --trans-nms 0.05 \
  --icp-dense --icp-dense-max 3000 \
  --render-rerank --render-score --mask-m 2n \
  --render-backend nvdiffrast \
  --out "$RUN/lmo/poses.csv" --cache "$RUN/lmo/cache" \
  --cand-csv "$RUN/lmo/cand.csv"

"$PY" examples/bop_eval.py \
  --bop "$BOP/ycbv" --dataset ycbv \
  --detections "$DET/cnos/cnos-fastsam_ycbv-test.json" \
  --merge none --topk 2 --grid 16 --solver gpu-feat --seed "$SEED" \
  --weights 1.0 \
  --use-s-coarse \
  --eq5-terms \
  --min-mask-pixels 0 --mask-iou-dedupe 1.1 \
  --tau-diameter \
  --trans-nms 0.05 \
  --icp-dense --icp-dense-max 3000 \
  --render-rerank --render-score --mask-m 2n \
  --render-backend nvdiffrast \
  --out "$RUN/ycbv/poses.csv" --cache "$RUN/ycbv/cache" \
  --cand-csv "$RUN/ycbv/cand.csv"
```

### faithful-4way

| Field | Frozen value |
|---|---|
| Report point | A / four-way |
| Code | **FROZEN** — tag `eighteen-run-freeze-20260807`; the operator supplies `POPOE_PIN=<its dereferenced full sha>` at run time (script refuses without it) and RUN_SPEC requires HEAD == PIN == tag^{commit}; the retired pin `twoline-rerank-fix-20260731` @ `509072e` predates the 2026-08-06 fix wave and identifies the voided runs only |
| Datasets | LM-O + YCB-V; one full BOP test run each |
| Detection inputs | CNOS + SAM6D (official 441) + NIDS + MUSE official JSONs under `data/detections/` — the four-source composition of the paper's rows-18/19 merged segmentation cell. The paper-style UNFILTERED union comes from `--min-mask-pixels 0` + `--mask-iou-dedupe 1.1` (cross-source masks are never deduped by design); `--merge none` separately disables the YCB-V clamp-pair label pooling (a no-op on LM-O) |
| Scoring | Paper Eq.7 three-term form with `--use-s-coarse` and `--eq5-terms` (both feature terms in the Eq.5 formulation: target->query top-k pool, fixed \|P_T^sparse\| denominator); Eq.7 exponents are **pinned-by-us** to unit exponents |
| Leaderboard comparator | A / four-way -> FreeZeV2.1(905) LM-O/YCB-V = 0.771 / 0.915 — **detection-matched** (same four sources; same *composition*, NOT the same files — see the MUSE vintage caveat in the §IV-A setup section); label the SAR + `M=2N` + render-scoring confounds. Paper Row 18 (75.9 / 91.3, four-way no SAR, self-reported, no leaderboard row) is auxiliary reference only |
| Artifacts | `$RUN/{lmo,ycbv}/` each contains `poses.csv`, `cand.csv`, `RECIPE.md`, `AR_SUMMARY.md`, `bop_server.md`, `grasp_summary.md` |
| Cautions | **Rerank + render score + M=2N carried (decision 13)**: the faithful arms align to v2.1 (Row 19), which carries SAR, M=2N and render scoring — popoe's `--render-rerank --render-score --mask-m 2n` are pinned-by-us approximations of those three (see the v2.1-only deltas table), disclosed as such, never as the official components. Dense resampling uses `rng(0)` where invoked and is independent of `--seed`. Encoding degradation is explicit in logs as `DEGRADE`. These runs are not bit/row comparable to historical anchors because seed and implementation fixes are new variables; the pre-decision-13 dev anchors (faithful 0.7314 / tuned 0.8174) are void — superseded by the 2026-08-07 re-anchor @ `efe97ab`: faithful-4way v4 **0.7369** / tuned-4way r3 **0.8243** AR(2/3) (LM-O obj-1 smoke, official-441 four-way). |

```bash
set -euo pipefail

POPOE="${POPOE:-/workspace/popoe}"
BOP="${BOP:-/workspace/bop_data}"
DET="${DET:-$POPOE/data/detections}"
: "${RUN_ROOT:?set RUN_ROOT=<fresh run root, e.g. /workspace/results/run18_20260807>}"
RUN="$RUN_ROOT/faithful-4way"
PY="${PY:-python}"
SEED=42

cd "$POPOE"
# Code identity: tag eighteen-run-freeze-20260807. The operator supplies its
# dereferenced full sha explicitly — never a bare tag name (stale-tag guard,
# RUN_SPEC). The old pin (twoline-rerank-fix-20260731 @ 509072e) predates the
# 2026-08-06 fix wave and identifies the voided runs only.
: "${POPOE_PIN:?set POPOE_PIN=<dereferenced full sha of tag eighteen-run-freeze-20260807>}"
if [ "$(git rev-parse HEAD)" != "$POPOE_PIN" ]; then
  echo "wrong popoe checkout; need POPOE_PIN=$POPOE_PIN (got HEAD=$(git rev-parse HEAD))" >&2
  exit 1
fi
# Rerank file guard — faithful re-carries rerank per decision 13 (2026-08-07):
for req in scripts/check_rerank_symmetry.py scripts/sar_render_compare.py; do
if [ ! -f "$req" ]; then
  echo "missing $req — stale tag tip (render_rerank loads sar_render_compare at RUNTIME); fetch --tags --force" >&2
  exit 1
fi
done
if [ -n "$(git status --porcelain)" ]; then
  echo "refusing dirty popoe worktree" >&2
  exit 1
fi

export BOP DET RUN SEED
export OMP_NUM_THREADS=8
export TORCH_HOME="${TORCH_HOME:-/workspace/torch_cache}"
export POPOE_GEDI_PATH=/workspace/gedi
export POPOE_BOP_TOOLKIT=/workspace/bop_toolkit
unset POPOE_TARGET_GRID POPOE_TARGET_CANON POPOE_TARGET_FILL POPOE_TARGET_CROP
unset POPOE_VIS_DIM POPOE_VIS_WEIGHT POPOE_SKIP_VIS POPOE_DINO_LAYER
unset POPOE_TWO_SCALE_GEDI POPOE_DGEDI_MODE POPOE_GEOM_BACKBONE POPOE_MESH_SHADING
unset POPOE_FPFH_RADII POPOE_FPFH_VOXEL_FRAC POPOE_FPFH_NORMAL_FRAC POPOE_FPFH_ORIENT
export POPOE_CANON_BASIS=diameter
export POPOE_QUERY_POINTS=5000
export POPOE_N_VIEWS=162
export POPOE_QUERY_CANON=476
export POPOE_QUERY_FILL=0.5
export POPOE_QUERY_FILL_MODE=effective
export POPOE_QUERY_MIN_VIEWS=18
export POPOE_QUERY_VIEWS=ico162
export POPOE_TARGET_DENSE=3000
export POPOE_TARGET_PAPER_GRID=1

mkdir -p "$RUN/lmo" "$RUN/ycbv"
for f in "$RUN/lmo/poses.csv" "$RUN/lmo/cand.csv" \
         "$RUN/ycbv/poses.csv" "$RUN/ycbv/cand.csv"; do
  # NOT `test ! -e a && test ! -e b`: under set -e a failing left-hand test
  # is errexit-exempt and the guard silently passes (second-review P0).
  if [ -e "$f" ]; then echo "refusing: $f exists — resume keeps old rows; use a FRESH --out" >&2; exit 1; fi
done

"$PY" examples/bop_eval.py \
  --bop "$BOP/lmo" --dataset lmo \
  --sources "cnos=$DET/cnos/cnos-fastsam_lmo-test.json,sam6d=$DET/sam6d/sam6d_official_lmo.json,nids=$DET/nids/nids_wa_sappe_lmo.json,muse=$DET/muse/muse-full_lmo-test.json" \
  --merge none --topk 2 --grid 16 --solver gpu-feat --seed "$SEED" \
  --weights 1.0 \
  --use-s-coarse \
  --eq5-terms \
  --min-mask-pixels 0 --mask-iou-dedupe 1.1 \
  --tau-diameter \
  --trans-nms 0.05 \
  --icp-dense --icp-dense-max 3000 \
  --render-rerank --render-score --mask-m 2n \
  --render-backend nvdiffrast \
  --out "$RUN/lmo/poses.csv" --cache "$RUN/lmo/cache" \
  --cand-csv "$RUN/lmo/cand.csv"

"$PY" examples/bop_eval.py \
  --bop "$BOP/ycbv" --dataset ycbv \
  --sources "cnos=$DET/cnos/cnos-fastsam_ycbv-test.json,sam6d=$DET/sam6d/sam6d_official_ycbv.json,nids=$DET/nids/nids_wa_sappe_ycbv.json,muse=$DET/muse/muse-full_ycbv-test.json" \
  --merge none --topk 2 --grid 16 --solver gpu-feat --seed "$SEED" \
  --weights 1.0 \
  --use-s-coarse \
  --eq5-terms \
  --min-mask-pixels 0 --mask-iou-dedupe 1.1 \
  --tau-diameter \
  --trans-nms 0.05 \
  --icp-dense --icp-dense-max 3000 \
  --render-rerank --render-score --mask-m 2n \
  --render-backend nvdiffrast \
  --out "$RUN/ycbv/poses.csv" --cache "$RUN/ycbv/cache" \
  --cand-csv "$RUN/ycbv/cand.csv"
```

### tuned-cnos

| Field | Frozen value |
|---|---|
| Report point | B / single-source |
| Code | **FROZEN** — tag `eighteen-run-freeze-20260807`; the operator supplies `POPOE_PIN=<its dereferenced full sha>` at run time (script refuses without it) and RUN_SPEC requires HEAD == PIN == tag^{commit}; the retired pin `twoline-rerank-fix-20260731` @ `509072e` predates the 2026-08-06 fix wave and identifies the voided runs only |
| Datasets | LM-O + YCB-V; one full BOP test run each |
| Detection inputs | CNOS only: `data/detections/cnos/cnos-fastsam_lmo-test.json`, `data/detections/cnos/cnos-fastsam_ycbv-test.json` |
| Scoring | Campaign2 tuned ChampionScorer: grid32, weights `1.0,0.7,0.5,0.3,0.2`; YCB-V uses `--merge ycbv --use-s-coarse`, LM-O uses `--merge none` and no `--use-s-coarse` |
| Leaderboard comparator | B / single-source -> FreeZe(CNOS) LM-O/YCB-V = 0.689 / 0.853 |
| Artifacts | `$RUN/{lmo,ycbv}/` each contains `poses.csv`, `cand.csv`, `RECIPE.md`, `AR_SUMMARY.md`, `bop_server.md`, `grasp_summary.md` |
| Cautions | Rerank scope differs from official SAR: popoe only reorders PCA flip variants. Dense resampling uses `rng(0)` where invoked and is independent of `--seed`. Encoding degradation is explicit in logs as `DEGRADE`. These runs are not bit/row comparable to historical anchors because seed, rerank and implementation fixes are new variables. |

```bash
set -euo pipefail

POPOE="${POPOE:-/workspace/popoe}"
BOP="${BOP:-/workspace/bop_data}"
DET="${DET:-$POPOE/data/detections}"
: "${RUN_ROOT:?set RUN_ROOT=<fresh run root, e.g. /workspace/results/run18_20260807>}"
RUN="$RUN_ROOT/tuned-cnos"
PY="${PY:-python}"
SEED=42

cd "$POPOE"
# Code identity: tag eighteen-run-freeze-20260807. The operator supplies its
# dereferenced full sha explicitly — never a bare tag name (stale-tag guard,
# RUN_SPEC). The old pin (twoline-rerank-fix-20260731 @ 509072e) predates the
# 2026-08-06 fix wave and identifies the voided runs only.
: "${POPOE_PIN:?set POPOE_PIN=<dereferenced full sha of tag eighteen-run-freeze-20260807>}"
if [ "$(git rev-parse HEAD)" != "$POPOE_PIN" ]; then
  echo "wrong popoe checkout; need POPOE_PIN=$POPOE_PIN (got HEAD=$(git rev-parse HEAD))" >&2
  exit 1
fi
for req in scripts/check_rerank_symmetry.py scripts/sar_render_compare.py; do
if [ ! -f "$req" ]; then
  echo "missing $req — stale tag tip (render_rerank loads sar_render_compare at RUNTIME); fetch --tags --force" >&2
  exit 1
fi
done
if [ -n "$(git status --porcelain)" ]; then
  echo "refusing dirty popoe worktree" >&2
  exit 1
fi

export BOP DET RUN SEED
export OMP_NUM_THREADS=16
export TORCH_HOME="${TORCH_HOME:-/workspace/torch_cache}"
export POPOE_GEDI_PATH=/workspace/gedi
export POPOE_BOP_TOOLKIT=/workspace/bop_toolkit
# Clear faithful pins that would otherwise leak from a prior arm in the same shell.
unset POPOE_CANON_BASIS POPOE_QUERY_POINTS POPOE_N_VIEWS POPOE_QUERY_CANON
unset POPOE_QUERY_FILL POPOE_QUERY_FILL_MODE POPOE_QUERY_MIN_VIEWS POPOE_QUERY_VIEWS
unset POPOE_TARGET_DENSE POPOE_TARGET_PAPER_GRID
unset POPOE_TARGET_GRID POPOE_TARGET_CANON POPOE_TARGET_FILL POPOE_TARGET_CROP
unset POPOE_VIS_DIM POPOE_VIS_WEIGHT POPOE_SKIP_VIS POPOE_DINO_LAYER
unset POPOE_TWO_SCALE_GEDI POPOE_DGEDI_MODE POPOE_GEOM_BACKBONE POPOE_MESH_SHADING
unset POPOE_FPFH_RADII POPOE_FPFH_VOXEL_FRAC POPOE_FPFH_NORMAL_FRAC POPOE_FPFH_ORIENT

mkdir -p "$RUN/lmo" "$RUN/ycbv"
for f in "$RUN/lmo/poses.csv" "$RUN/lmo/cand.csv" \
         "$RUN/ycbv/poses.csv" "$RUN/ycbv/cand.csv"; do
  # NOT `test ! -e a && test ! -e b`: under set -e a failing left-hand test
  # is errexit-exempt and the guard silently passes (second-review P0).
  if [ -e "$f" ]; then echo "refusing: $f exists — resume keeps old rows; use a FRESH --out" >&2; exit 1; fi
done

"$PY" examples/bop_eval.py \
  --bop "$BOP/lmo" --dataset lmo \
  --detections "$DET/cnos/cnos-fastsam_lmo-test.json" \
  --merge none --topk 2 --grid 32 --solver o3d --seed "$SEED" \
  --weights 1.0,0.7,0.5,0.3,0.2 \
  --trans-nms 0.05 \
  --render-rerank --render-score --mask-m 2n \
  --render-backend nvdiffrast \
  --out "$RUN/lmo/poses.csv" --cache "$RUN/lmo/cache" \
  --cand-csv "$RUN/lmo/cand.csv"

"$PY" examples/bop_eval.py \
  --bop "$BOP/ycbv" --dataset ycbv \
  --detections "$DET/cnos/cnos-fastsam_ycbv-test.json" \
  --merge ycbv --topk 2 --grid 32 --solver o3d --seed "$SEED" \
  --weights 1.0,0.7,0.5,0.3,0.2 \
  --use-s-coarse \
  --trans-nms 0.05 \
  --render-rerank --render-score --mask-m 2n \
  --render-backend nvdiffrast \
  --out "$RUN/ycbv/poses.csv" --cache "$RUN/ycbv/cache" \
  --cand-csv "$RUN/ycbv/cand.csv"
```

### tuned-4way

| Field | Frozen value |
|---|---|
| Report point | B / four-way |
| Code | **FROZEN** — tag `eighteen-run-freeze-20260807`; the operator supplies `POPOE_PIN=<its dereferenced full sha>` at run time (script refuses without it) and RUN_SPEC requires HEAD == PIN == tag^{commit}; the retired pin `twoline-rerank-fix-20260731` @ `509072e` predates the 2026-08-06 fix wave and identifies the voided runs only |
| Datasets | LM-O + YCB-V; one full BOP test run each |
| Detection inputs | CNOS + SAM6D + NIDS + official MUSE JSONs under `data/detections/`; `muse` means downloaded official artefacts, not `muse-repro` |
| Scoring | Campaign2 tuned ChampionScorer: grid32, weights `1.0,0.7,0.5,0.3,0.2`; YCB-V uses `--merge ycbv --use-s-coarse`, LM-O uses `--merge none` and no `--use-s-coarse` |
| Leaderboard comparator | B / four-way -> FreeZeV2.1(905) LM-O/YCB-V = 0.771 / 0.915 — detection-matched (same four sources; same *composition*, NOT the same files — see the MUSE vintage caveat in the §IV-A setup section); label the SAR + `M=2N` + render-scoring confounds. A/four-way -> B/four-way is the clean improvement-package column (identical detection inputs) |
| Artifacts | `$RUN/{lmo,ycbv}/` each contains `poses.csv`, `cand.csv`, `RECIPE.md`, `AR_SUMMARY.md`, `bop_server.md`, `grasp_summary.md` |
| Cautions | Rerank scope differs from official SAR: popoe only reorders PCA flip variants. Dense resampling uses `rng(0)` where invoked and is independent of `--seed`. Encoding degradation is explicit in logs as `DEGRADE`. These runs are not bit/row comparable to historical anchors because seed, rerank and implementation fixes are new variables. |

```bash
set -euo pipefail

POPOE="${POPOE:-/workspace/popoe}"
BOP="${BOP:-/workspace/bop_data}"
DET="${DET:-$POPOE/data/detections}"
: "${RUN_ROOT:?set RUN_ROOT=<fresh run root, e.g. /workspace/results/run18_20260807>}"
RUN="$RUN_ROOT/tuned-4way"
PY="${PY:-python}"
SEED=42

cd "$POPOE"
# Code identity: tag eighteen-run-freeze-20260807. The operator supplies its
# dereferenced full sha explicitly — never a bare tag name (stale-tag guard,
# RUN_SPEC). The old pin (twoline-rerank-fix-20260731 @ 509072e) predates the
# 2026-08-06 fix wave and identifies the voided runs only.
: "${POPOE_PIN:?set POPOE_PIN=<dereferenced full sha of tag eighteen-run-freeze-20260807>}"
if [ "$(git rev-parse HEAD)" != "$POPOE_PIN" ]; then
  echo "wrong popoe checkout; need POPOE_PIN=$POPOE_PIN (got HEAD=$(git rev-parse HEAD))" >&2
  exit 1
fi
for req in scripts/check_rerank_symmetry.py scripts/sar_render_compare.py; do
if [ ! -f "$req" ]; then
  echo "missing $req — stale tag tip (render_rerank loads sar_render_compare at RUNTIME); fetch --tags --force" >&2
  exit 1
fi
done
if [ -n "$(git status --porcelain)" ]; then
  echo "refusing dirty popoe worktree" >&2
  exit 1
fi

export BOP DET RUN SEED
export OMP_NUM_THREADS=16
export TORCH_HOME="${TORCH_HOME:-/workspace/torch_cache}"
export POPOE_GEDI_PATH=/workspace/gedi
export POPOE_BOP_TOOLKIT=/workspace/bop_toolkit
# Clear faithful pins that would otherwise leak from a prior arm in the same shell.
unset POPOE_CANON_BASIS POPOE_QUERY_POINTS POPOE_N_VIEWS POPOE_QUERY_CANON
unset POPOE_QUERY_FILL POPOE_QUERY_FILL_MODE POPOE_QUERY_MIN_VIEWS POPOE_QUERY_VIEWS
unset POPOE_TARGET_DENSE POPOE_TARGET_PAPER_GRID
unset POPOE_TARGET_GRID POPOE_TARGET_CANON POPOE_TARGET_FILL POPOE_TARGET_CROP
unset POPOE_VIS_DIM POPOE_VIS_WEIGHT POPOE_SKIP_VIS POPOE_DINO_LAYER
unset POPOE_TWO_SCALE_GEDI POPOE_DGEDI_MODE POPOE_GEOM_BACKBONE POPOE_MESH_SHADING
unset POPOE_FPFH_RADII POPOE_FPFH_VOXEL_FRAC POPOE_FPFH_NORMAL_FRAC POPOE_FPFH_ORIENT

mkdir -p "$RUN/lmo" "$RUN/ycbv"
for f in "$RUN/lmo/poses.csv" "$RUN/lmo/cand.csv" \
         "$RUN/ycbv/poses.csv" "$RUN/ycbv/cand.csv"; do
  # NOT `test ! -e a && test ! -e b`: under set -e a failing left-hand test
  # is errexit-exempt and the guard silently passes (second-review P0).
  if [ -e "$f" ]; then echo "refusing: $f exists — resume keeps old rows; use a FRESH --out" >&2; exit 1; fi
done

"$PY" examples/bop_eval.py \
  --bop "$BOP/lmo" --dataset lmo \
  --sources "cnos=$DET/cnos/cnos-fastsam_lmo-test.json,sam6d=$DET/sam6d/sam6d_official_lmo.json,nids=$DET/nids/nids_wa_sappe_lmo.json,muse=$DET/muse/muse-full_lmo-test.json" \
  --merge none --topk 2 --grid 32 --solver o3d --seed "$SEED" \
  --weights 1.0,0.7,0.5,0.3,0.2 \
  --trans-nms 0.05 \
  --render-rerank --render-score --mask-m 2n \
  --render-backend nvdiffrast \
  --out "$RUN/lmo/poses.csv" --cache "$RUN/lmo/cache" \
  --cand-csv "$RUN/lmo/cand.csv"

"$PY" examples/bop_eval.py \
  --bop "$BOP/ycbv" --dataset ycbv \
  --sources "cnos=$DET/cnos/cnos-fastsam_ycbv-test.json,sam6d=$DET/sam6d/sam6d_official_ycbv.json,nids=$DET/nids/nids_wa_sappe_ycbv.json,muse=$DET/muse/muse-full_ycbv-test.json" \
  --merge ycbv --topk 2 --grid 32 --solver o3d --seed "$SEED" \
  --weights 1.0,0.7,0.5,0.3,0.2 \
  --use-s-coarse \
  --trans-nms 0.05 \
  --render-rerank --render-score --mask-m 2n \
  --render-backend nvdiffrast \
  --out "$RUN/ycbv/poses.csv" --cache "$RUN/ycbv/cache" \
  --cand-csv "$RUN/ycbv/cand.csv"
```

## Headline ledger

> **2026-07-26 pipeline verify COMPLETE** — popoe **`75553a1`**.
> Artifacts: `outputs/pipeline_verify_20260726/`. Full AR = mean(MSSD, MSPD, VSD).  
> **Δ exceeds ±0.003** on full AR; all six rows land **above** the historical script-line figures. Do **not** rewrite those headlines; cite this as the popoe parity measurement (dual-track). Promotion line remains 0.8201 / 0.6896.
>
> **Calibre note (2026-07-29, post PR #23)**: every locally scored AR in this ledger is **legacy per-object calibre** (the pre-#23 scorer averaged objects with equal weight). BOP flat (per-instance) calibre, recomputed from the same CSVs: **#1 = 0.7892** (vs 0.7781), **#2 = 0.6844** (vs 0.6792); the historical script-line figures in flat calibre are 0.7766 / 0.6777. The promotion-line figures above are also legacy calibre (fourway flat: 0.8444 / 0.7106). Grasp rows #5/#6 use per-object-median statistics, unchanged. From #23 onward `popoe.metrics` reports flat calibre by default (legacy value kept as a trailing diagnostic line).

| # | Experiment | Archive number (source) | popoe entrypoint | Class | Reproduced | popoe commit / pod / date | Status |
|---|---|---|---|---|---|---|---|
| 1 | YCB-V full BOP AR | **0.7668** — `score_rules_ycbvg32m`; recipe: CNOS-FastSAM TOPK2 + gripper label pooling + grid-32 + O3D + fit×s_feat_1(×metric) | `examples/bop_eval.py --bop $BOP/ycbv --detections data/detections/cnos/cnos-fastsam_ycbv-test.json --merge ycbv --topk 2 --grid 32 --solver o3d --weights 1.0,0.7,0.5,0.3,0.2 --render-backend nvdiffrast --out … --cache … --cand-csv …` | **GPU-POD** | **0.7781** (MSSD 0.7934 / MSPD 0.7414 / VSD 0.7995) | `75553a1` / 2026-07-26 | ☑ Δ=+0.0113 |
| 2 | LM-O full BOP AR | **0.6726** — `lmog32`; CNOS∪SAM6D union detections + same pipeline | `examples/bop_eval.py --bop $BOP/lmo --sources cnos=…/cnos-fastsam_lmo-test.json,sam6d=…/sam6d_ism_lmo.json --merge none --topk 2 --grid 32 --solver o3d …` | **GPU-POD** | **0.6792** (MSSD 0.7242 / MSPD 0.7566 / VSD 0.5568) | same campaign | ☑ Δ=+0.0066 |
| 3 | YCB-V AR(2/3) | 0.7528 (same run as #1) | same pose CSV as #1; score with `python -m popoe.metrics.ar` | **LOCAL-CPU** (post #1) | **0.7674** | same | ☑ Δ=+0.0146 |
| 4 | LM-O AR(2/3) | 0.7324 (same run as #2) | same pose CSV as #2; same AR scorer as #3 | **LOCAL-CPU** (post #2) | **0.7404** | same | ☑ Δ=+0.0080 |
| 5 | YCB-V grasp ADD(-S)@0.1d | **0.8173** (median 2.5 mm / 6.6°) | `python -m popoe.metrics.grasp` on #1 CSV | **LOCAL-CPU** (post #1) | **0.8240** (@0.05d 0.7716; med 2.5 mm / 11.9°) | same | ☑ Δ=+0.0067 |
| 6 | LM-O grasp ADD(-S)@0.1d | **0.7617** (7.2 mm / 5.8°) | same as #5 on #2 CSV | **LOCAL-CPU** (post #2) | **0.7706** (@0.05d 0.5146; med 7.1 mm / 5.9°) | same | ☑ Δ=+0.0089 |

## BOP-Classic-Core seven-set ledger (2026-07-27, OFFICIAL SERVER SCORES)

> The seven core datasets scored by the **BOP evaluation server**, method
> `popoe-cnos`, popoe **`0c93d3e`**.
> Submissions 39689–39695, all kept **private**. These are not local
> measurements — the server computed them from the uploaded pose CSVs.
> Recipe: official CNOS-FastSAM default detections, **single source** (same
> footing as the official FreeZe(CNOS) row); `--merge ycbv` on YCB-V, `auto`
> elsewhere.

| Dataset | AR | AR_MSSD | AR_MSPD | AR_VSD | FreeZe(CNOS) | Δ | s/image |
|---|---:|---:|---:|---:|---:|---:|---:|
| TUD-L | 0.902 | 0.924 | 0.925 | 0.859 | 0.936 | −3.4pt | 2.52 |
| YCB-V | 0.787 | 0.797 | 0.746 | 0.818 | 0.853 | −6.6pt | 3.61 |
| HB | 0.690 | 0.685 | 0.685 | 0.701 | 0.790 | −10.0pt | 10.07 |
| LM-O | 0.631 | 0.670 | 0.700 | 0.523 | 0.689 | −5.8pt | 6.29 |
| ITODD | **0.565** | 0.607 | 0.592 | 0.495 | 0.561 | **+0.4pt** | 5.71 |
| T-LESS | 0.443 | 0.462 | 0.477 | 0.388 | 0.520 | −7.7pt | 12.96 |
| IC-BIN | 0.387 | 0.343 | 0.309 | 0.510 | 0.499 | −11.2pt | 35.44 |
| **ARCore** | **0.6293** | | | | **0.6926** | **−6.3pt** | |

**Pipeline self-validation**: YCB-V's server AR 0.787 agrees with headline
row #1 under an identical recipe. Row #1's ledger value 0.7781 is legacy
per-object calibre; re-aggregated in the server's flat calibre it is
**0.7892**, so the genuine run-to-run difference is **+0.2pt** (RANSAC-noise
scale). The +0.9pt previously quoted here compared across calibres — about
1pt of it was aggregation, not noise (see PR #23). The end-to-end chain is
sound, so every gap above is methodological, not a defect.

**What the gaps are not caused by**: the official FreeZe(CNOS) row consumes
the *same* CNOS-FastSAM detection file, so every per-dataset gap arises in
the pose stage (registration + scoring), not in detection.

**The per-dataset spread is NOT yet explained.** An earlier version of this
section claimed it tracked multi-instance-ness and blamed
`adapters.select_top_instances` for lacking spatial de-duplication. **That
claim is withdrawn — the submitted CSVs refute it:**

- ITODD has the *most* multi-instance targets of the seven (75.7%, up to 8+
  instances) and is the only dataset that beats the official row; HB is
  100% single-instance and sits at −10.0pt. Single-instance gaps
  (−3.4/−5.8/−6.6/−10.0) and multi-instance gaps (+0.4/−7.7/−11.2) have
  essentially the same mean.
- Champions within one target are not stacking on one physical instance:
  median pairwise translation is 236mm (IC-BIN) / 99mm (T-LESS) / 97mm
  (ITODD), and only 0.3–0.9% of champion pairs sit closer than 5mm.

A second candidate died too: the winning visual weight (per-target argmax
over `--weights`) is distributed almost identically across all seven sets
(mean 0.56–0.65), so "our visual branch is weaker than theirs" has no
support either.

Settling this needs GT-based per-instance analysis — separating "wrong mask
or wrong object" from "right mask, poor registration" — which requires the
GT trees on the pod volume for the five datasets not held locally. Until
then the spread is recorded as unexplained rather than narrated.

> **2026-07-28 — every LM-O number above and below predates the shading fix.**
> LM-O carries its colour in `property uchar red/green/blue` with no UV atlas.
> Until popoe `b439d58`, such meshes were classified as untextured and rendered
> flat beige, so the DINOv2 half of every LM-O query feature was computed on a
> colourless image. This applies to **headline rows 2, 4 and 6** and to the
> **LM-O row of the seven-set table** (0.631). Re-running those commands at or
> after that commit will **not** reproduce their numbers, and should not: the
> measured move is **+1.31 pt** full AR (0.6876 → 0.7007) — on the
> **CNOS∪SAM6D union line**; an earlier version of this note said "seven-set
> CNOS line", which its own numbers contradict (that line's row is 0.631).
> On the seven-set CNOS single-source line the same fix measures **+1.07 pt**
> (0.6306 → 0.6413, `corrtopk_20260728/lmo_base`). Quote per line.
>
> Three more sets in the seven-set table are vertex-coloured and carry the same
> caveat, **unmeasured**: TUD-L, IC-BIN, HB. YCB-V (rows 1/3/5 and its
> seven-set row) is unaffected — it ships a real UV atlas. T-LESS
> (`models_cad`) and ITODD have no colour at all, so flat beige was already
> the correct render for them; ITODD is also the only set we beat official on.
> The affected and unaffected sets interleave in gap size, so this fix does
> **not** explain the per-dataset spread discussed above.
>

## Contribution-level parity (secondary)

| Experiment | Archive result | popoe entrypoint | Class | Status |
|---|---|---|---|---|
| Adaptive visual weight | beats best-fixed on all 4 datasets | Built into `bop_eval.py --weights 1.0,0.7,0.5,0.3,0.2` (ChampionScorer per-target argmax over w). Cross-dataset 4-set claim still needs TUD-L / IC-BIN BOP data + GPU runs (not in this repo). | **GPU-POD** (YCB-V/LM-O covered by #1/#2); **GAP** for TUD-L/IC-BIN data | ☐ |
| Canonical-space scoring | 26-rule ablation; champion rule constant across datasets | Live rule = `ChampionScorer` (`s_icp * max(s_feat_1,0) * metric_fit?`). Offline re-sweep: `examples/rule_replay.py <cand.csv> --target-csv <poses.csv> --rule "s_icp*s_feat_1" --rule "s_icp*s_feat_1*metric_fit" --out-dir …` on a `--cand-csv` dump from #1/#2. `--target-csv` defines the full target universe and zero-pads detector misses; without it, output AR is a candidate-bearing ceiling. | **LOCAL-CPU** (once cand-csv exists) | ☐ |
| Gripper label pooling + metric_fit | obj20 +33.6 pt; 2×2 ablation | `bop_eval.py --merge ycbv` (pools 19:20, size_aware metric_fit) vs `--merge none` on YCB-V objs 19,20 (`--objs 19,20`). Live scorer: `ChampionScorer(size_aware=True)` for pooled pairs. | **GPU-POD** (subset ablation) | ☐ |
| Multi-mask / detection union (LM-O) | CNOS∪SAM6D +2.8 pt | Smoke (no GPU): `examples/union_smoke.py --dataset lmo --source sam6d=data/detections/sam6d/sam6d_ism_lmo.json`. Full AR: same as headline #2 (`--sources cnos=…,sam6d=…`). CNOS-only control: `--detections …/cnos-fastsam_lmo-test.json`. | **LOCAL-CPU** smoke + **GPU-POD** full | ☑ full-AR via #2 (2026-07-26); no separate CNOS-only control re-run |

## Ledger rules

1. Record the popoe commit hash for every reproduced number. Code used for
   a cited run is a `git clone` at that commit, not a copied file.
2. Raw per-image CSVs stay out of git. This file records the number,
   commit, and artefact location.
3. One row per run: if a re-run disagrees beyond tolerance, add a row;
   do not overwrite.

## Segmentation AP ledger (2026-07-26)

> **Status**: official-source offline measurements from
> `outputs/seg_ap_20260725T223014Z/`; `muse-repro` G3 rows from the
> 2026-07-26 default-promotion campaign.
> **Not** a pose parity row. Pose headline #1/#2 filled above from the same day's campaign.  
> PRs #3/#4/#5/#6 are merged; evaluator semantics from #5 apply to future re-scores.

### Official single-source mask AP (YCB-V / LM-O)

Local evaluator: popoe `examples/bop_seg_eval.py` + PyPI `pycocotools 2.0.11`,
except where noted. Public rows from BOP segmentation-unseen leaderboards.
Detail: `outputs/seg_ap_20260725T223014Z/LEADERBOARD_ALIGNMENT.md`.

| Dataset | Source tag | Local AP | Public AP | Δ AP | Verdict |
|---|---|---:|---:|---:|---|
| YCB-V | `cnos` (CNOS-FastSAM JSON) | 0.5986 | 0.5987 | −0.0001 | **aligned** |
| YCB-V | `nids` (NIDS-Net WA_Sappe) | 0.6499 | 0.6500 | −0.0001 | **aligned** |
| YCB-V | `sam6d` (local ISM file) | 0.6112 | (see note) | — | file-aligned; not identical to every BOP SAM6D row |
| YCB-V | `muse` (official BOP sub 29113) | 0.6901 | 0.6900 | +0.0001 | **aligned** (official artefact) |
| LM-O | `cnos` | 0.3921 | 0.3969 | −0.0048 | **bracketed** by pycocotools vs BOP COCO fork (~±0.005) |
| LM-O | `nids` | 0.4345 | 0.4393 | −0.0048 | same evaluator residual |
| LM-O | `sam6d` (local ISM) | 0.4411 | — | — | local file |
| LM-O | `muse` (official BOP sub 29108) | 0.4713 | 0.4770 | −0.0057 | same residual class |

### `muse-repro` G3 AP (default-promotion evidence)

These are **reimplementation** rows (`source='muse-repro'`), not official
`muse` artefacts and not a pose-promotion line. Recipe:
`--mask-rgb --gem-tokens all`, default depth gate, default similarity
`(class_sim=cosine, patch_sim=tanimoto)`, full test split.

| Dataset | Official `muse` AP | Pre-G3 `muse-repro` AP | G3 `muse-repro` AP | Δ vs official | popoe commit | Artefacts | Verdict |
|---|---:|---:|---:|---:|---|---|---|
| LM-O | 0.471 | 0.228 | **0.388** | −0.083 | `e57cf03` | `outputs/g3_muse_mask_rgb_20260726/` | recovers most of the AP gap; residual remains |
| YCB-V | 0.690 | 0.326 | **0.684** | −0.006 | `b8614c1` | `outputs/g3_muse_ycbv_mask_rgb_20260726/` | parity-level with official |

PR #10 promotes this G3 recipe as the default for `build_muse_segmentor`,
`popoe-muse`, and `popoe-bop-muse`. Historical reproduction remains available
with `--no-mask-rgb --gem-tokens fg`.

With BOP toolkit's declared COCO fork, YCB-V CNOS/NIDS match public AP to
machine precision; LM-O sits ~0.006 **above** public (ignore-threshold
sensitivity — see `LEADERBOARD_ALIGNMENT.md` ignore sweep).

### Naming (do not mix)

| `Detection.source` | Meaning |
|---|---|
| `cnos` / `sam6d` / `nids` / `muse` | Official or precomputed JSON. **Nothing in popoe writes `muse`.** |
| `muse-repro` | `popoe.segmentor_muse` / `popoe-bop-muse` reimplementation |
| `cnos-lab` | Self-built CNOS lab track (formerly `cnos-v3`) — not paper headline |

Four-way pose still uses official `muse` JSON, not `muse-repro`.

### Human review exporter

```bash
uv run python scripts/export_bop_seg_review.py \
  --bop bop_data/ycbv \
  --source cnos=data/detections/cnos/cnos-fastsam_ycbv-test.json \
  --source nids=data/detections/nids/nids_wa_sappe_ycbv.json \
  --source sam6d=data/detections/sam6d/sam6d_ism_ycbv.json \
  --source muse=outputs/seg_ap_20260725T223014Z/official_submissions/muse-full_ycbv-test_official.json \
  --out-dir outputs/pipeline_verify_seg_vis/ycbv \
  --per-source 8 --worst-per-obj 1 --topk 1 --seed 0
```

Produces per-source `*_overlay.png` / `*_mask.png` / `*_crop.png` plus
`INDEX.md` (IoU vs `mask_visib` when GT is on disk).

## Solver A/B ledger (2026-07-26)

**Not a parity row and not a performance claim.** This is the sole home of the
solver-swap ranking: three `PoseSolver` implementations against *each other*
through one identical chain. [ARCHITECTURE.md](ARCHITECTURE.md#pluggability-proven--the-posesolver-stage)
describes the seam; it does not quote these numbers. `solver_swap_demo` is not
the evaluated pipeline (it runs `FreeZeScorer` on GT masks at fixed thresholds)
and obj 5 is a known-weak registration case — `recall@0.1d` is 0.000 for all
three. An older rotation-angle table was withdrawn; cite these MSSD numbers
only (see ISSUES.md). Metric: bop_toolkit `pose_error.mssd`, symmetries from
`models_eval/models_info.json` (obj 5 is identity-only). Seeded,
`--seed 42`. 140 of 150 test instances (scene 52 incomplete).

| Solver | median MSSD | @0.2d | @0.5d | popoe commit | Artefacts | Verdict |
|---|---:|---:|---:|---|---|---|
| `RansacSolver` (freeze_ransac) | **42.9 mm** (0.218 d) | **0.371** | **0.600** | `63c5e7d` | `outputs/solver_swap_20260726/` | best of the three |
| `Open3DFeatureRansacSolver` 1-shot | 111.2 mm (0.566 d) | 0.271 | 0.457 | same | same | baseline for the composition |
| `Open3DFeatureRansacSolver(n_restarts=8)` | 62.7 mm (0.319 d) | 0.343 | 0.521 | same | same | composition helps, parity NOT reached |

Reading: handing several hypotheses to the scorer nearly halves 1-shot's median
MSSD and wins head-to-head 72:33 (35 tied), closing roughly two thirds of the
gap to `freeze_ransac` — but not reaching it. Ordering is stable at every
threshold.

Artefacts: `outputs/solver_swap_20260726/` (gitignored). Figures were
recomputed from the 140 per-instance rows of `mssd140_run.log`.

## Unified 18-run freeze

Tag `eighteen-run-freeze-20260807`. Arms: `faithful-cnos` and `faithful-4way`
on LM-O+YCB-V; `tuned-cnos` and `tuned-4way` on all seven BOP-Classic-Core
sets. Detection inputs are official BOP artefacts; SHA256 pins are in the
tables below and in `data/detections/*/PROVENANCE.md`.

Every run supplies the tag's dereferenced full sha as `POPOE_PIN` and
records it in `RECIPE.md`. Decision-13 dev anchors: faithful-4way v4
0.7369 / tuned-4way r3 0.8243 AR(2/3) on LM-O obj-1 smoke. Every arm
passes `--trans-nms 0.05`
— paper §III-F translation NMS on refined poses; the paper names the mechanism
but no radius, so the value (0.05× the models_info diameter; same-instance duplicates converge post-ICP within ~1-2% of the diameter, and nested/thin objects can hold distinct instances closer than 0.1) is pinned-by-us
and parameterised. Default-on in `bop_eval.py`; the four runbooks spell it
explicitly so a future default change cannot silently re-identity the
freezes. `--trans-nms 0` disables and must be recorded as a deviation.

**Server-only acceptance for the four itodd/hb runs.** Their test GT
is withheld by BOP, so `ar_flat.py` has nothing to score and there is NO local
AR gate — deciding what to do about that mid-run is how accidents happen, so
the criteria are fixed here in advance. Acceptance for E-cnos/itodd, E-cnos/hb,
E-4way/itodd, E-4way/hb is: (a) the row-count completion invariant holds
exactly; (b) DEGRADE counts and per-source candidate counts are the same order
as the neighbouring sets' runs; (c) submit private to the BOP server directly —
the server return is the ONLY score. Do not wire `--probe-corr` there (needs
GT; preflight item 6). Diagnosis on these sets uses the val splits (public GT,
downloaded 2026-08-07) and is a post-run activity, never an acceptance gate.

Detection inputs are real bytes (no symlinks, file or directory) bound to
their paths by `data/detections/MANIFEST.sha256` (tracked; path+hash — a
registered file at the wrong path fails) and registered in the per-source
`PROVENANCE.md`; `scripts/freeze_detections.py` enforces both. Pods run it
before any eval as `--check --need <the run's relative paths>` — `--need`
is the per-run completeness gate (a lost rsync fails loudly instead of
reading as an empty-but-clean tree).

**Preflight — per host, per dataset, before any full run:**

1. **Code identity**: fresh clone; `POPOE_PIN` equality; clean worktree;
   `python -c "import popoe; print(popoe.__file__)"` prints the pinned
   clone's path.
2. **Inputs**: `scripts/freeze_detections.py --check --need <paths>`.
3. **Stale outputs**: the run's `--out` / `--cand-csv` must not exist
   (scripts refuse). Resume classifies by row count and would silently
   "complete" on a leftover CSV. Move old CSVs aside first.
4. **Source ingestion**: after the per-set smoke, every wired source
   must have a nonzero count in `cand.csv`. An empty source passes
   schema checks and just loses recall.
5. **New-set layout**: before a full run on tless / itodd / hb / icbin /
   tudl, load one image end-to-end (`--objs <one id>`): checks models dir
   naming (tless `models_cad`), depth
   format, and the itodd grayscale 16-bit tif path (the visual branch
   hard-fails on non-uint8 input by design).
6. **No local score on itodd/hb**: their test GT is withheld, so
   `ar_flat.py` cannot score. The only score gate is the BOP server.
   Do not wire `--probe-corr` there (needs GT).
7. **Merge spelling**: `--merge ycbv` exists only on the tuned YCB-V
   leg; every other (set, arm) runs `--merge none`.
8. **Rerank sanity** (all arms): the run log must show `[rerank] y_sign
   latched` with a healthy IoU before bulk targets; repeating
   `UNRELIABLE` means the renderer is miscalibrated — stop.

Common pins (both lines): `--topk 2 --grid 32 --solver o3d --seed 42
--weights 1.0,0.7,0.5,0.3,0.2 --render-rerank --render-score --mask-m 2n
--render-backend nvdiffrast`;
YCB-V uses `--merge ycbv --use-s-coarse`; all other datasets `--merge none`,
no `--use-s-coarse`. OMP/host/sharding are runtime provenance only.

### E-cnos (`tuned-cnos` x 7)

Comparator: FreeZe(CNOS) per-set (same detection input). LM-O + YCB-V were to
serve as Phase D B-single results as well: under the unified 18-run table they
are the same runs, not a later reuse or a second identity. All seven rows are
run under the new frozen pin.

| Dataset | Detection input | SHA256 |
|---|---|---|
| lmo | `data/detections/cnos/cnos-fastsam_lmo-test.json` | `1a03d3c7a1d57a9c7e6e1bc162f99281b5044ca50428c619477ec4ab11fa375a` |
| tudl | `data/detections/cnos/cnos-fastsam_tudl-test.json` | `400978b21a94aaa109d6e5039df7aefa7cdbdc6af037cbd1b05ab586ae6d540d` |
| tless | `data/detections/cnos/cnos-fastsam_tless-test.json` | `db010fbce92149a54ae7a252176d6dee80823353a7e5d704c0f33657c5b1ecec` |
| icbin | `data/detections/cnos/cnos-fastsam_icbin-test.json` | `922b9878b1e8e8cac7d9245daa672de7568408ca0d4a8f9a7884bb532f93bcc3` |
| itodd | `data/detections/cnos/cnos-fastsam_itodd-test.json` | `cce4bcc9d33618e215f1099f9ac7f04598c0f39188585e739dd992496c3bbbd6` |
| hb | `data/detections/cnos/cnos-fastsam_hb-test.json` | `7eb39ad0d82783dc59a49cd2f6654c99b63d3b3ef3f051f3368056755e94e6b0` |
| ycbv | `data/detections/cnos/cnos-fastsam_ycbv-test.json` | `fdec15729676e15876302fc620f752cc5290ee28da5fc3c7e17da1072fd4f422` |

### E-4way (`tuned-4way`-official x 7)

Comparator: FreeZeV2.1(905) per-set — confounded by SAR + M=2N +
render-scoring, label every reading; secondary 756. **SAM6D = official BOP
submissions for ALL SEVEN sets, method 441 "SAM6D"** (441 is the
strongest official SAM6D variant, mean seg AP 0.481 vs 546's 0.428, board
family spread 5.3pt; seg batch 6965-6971, 2023-12-05; 441's method page mixes
THREE tasks — seg, 2D detection, and 6D localization batches — the seg batch
was identified by matching per-set AP against the leaderboard row). The
LM-O/YCB-V rows are simultaneously Phase D B-four and Phase E E-4way rows;
there is no local-ISM Phase D identity in the unified table.

| Dataset | Source | Detection input | SHA256 |
|---|---|---|---|
| lmo | cnos | `data/detections/cnos/cnos-fastsam_lmo-test.json` | `1a03d3c7a1d57a9c7e6e1bc162f99281b5044ca50428c619477ec4ab11fa375a` |
| lmo | sam6d | `data/detections/sam6d/sam6d_official_lmo.json` | `638a933c0f3f404086f975050524ead00b23f6c081d77a1dce99443fab781108` |
| lmo | nids | `data/detections/nids/nids_wa_sappe_lmo.json` | `8cf9c392a82153b3bbf1c6baa5a7a4fac056e6fc4f35ec645a1f3f76d6f75aea` |
| lmo | muse | `data/detections/muse/muse-full_lmo-test.json` | `55061983089d6236c19cb9b6a8a6c754388d146287be45ec40ceb9c32dbe3003` |
| tudl | cnos | `data/detections/cnos/cnos-fastsam_tudl-test.json` | `400978b21a94aaa109d6e5039df7aefa7cdbdc6af037cbd1b05ab586ae6d540d` |
| tudl | sam6d | `data/detections/sam6d/sam6d_official_tudl.json` | `267784437d15d97061dc30248bacdb08631780385fc7b86507818fd7ef63a6ab` |
| tudl | nids | `data/detections/nids/nids_wa_sappe_tudl.json` | `90137dcec2f140d2b8130e72524d751d1b94fd751efd90a05c8a089861357c4e` |
| tudl | muse | `data/detections/muse/muse-full_tudl-test.json` | `38dc40cfa75f22a74f1f85cb10fb2283adb99db65ea496dff68bf216beeccb8b` |
| tless | cnos | `data/detections/cnos/cnos-fastsam_tless-test.json` | `db010fbce92149a54ae7a252176d6dee80823353a7e5d704c0f33657c5b1ecec` |
| tless | sam6d | `data/detections/sam6d/sam6d_official_tless.json` | `e63e91376d3c116ea39aec2b5c173b0358f099fbb260275ab69fe48200e6fdf6` |
| tless | nids | `data/detections/nids/nids_wa_sappe_tless.json` | `16da4f7965e3adcaaa432163ba9f2953d42a3987aca1f38e7dcc42295901b11b` |
| tless | muse | `data/detections/muse/muse-full_tless-test.json` | `78bcdab72d0eac44ab5b8477eec9e229fdaa2e61fdc69bcec48be46a3f230482` |
| icbin | cnos | `data/detections/cnos/cnos-fastsam_icbin-test.json` | `922b9878b1e8e8cac7d9245daa672de7568408ca0d4a8f9a7884bb532f93bcc3` |
| icbin | sam6d | `data/detections/sam6d/sam6d_official_icbin.json` | `3e4797bfda1dc2ca7514018ed082b6c8678351bf6905f2803725825bc167cef8` |
| icbin | nids | `data/detections/nids/nids_wa_sappe_icbin.json` | `2a39dad6d5273c45ef6c88415a78f30e7e6819bb654210a0917b8dcc1ca580cd` |
| icbin | muse | `data/detections/muse/muse-full_icbin-test.json` | `34a2a40b3c716bb3c36b0739d49ebc885019cb6d97079d4e5e1ba9c743ed1427` |
| itodd | cnos | `data/detections/cnos/cnos-fastsam_itodd-test.json` | `cce4bcc9d33618e215f1099f9ac7f04598c0f39188585e739dd992496c3bbbd6` |
| itodd | sam6d | `data/detections/sam6d/sam6d_official_itodd.json` | `d0511f138d0e509ee3fb028e5d3c438fa1f2cb6ae27e1ef4fc6d22b3968595e2` |
| itodd | nids | `data/detections/nids/nids_wa_sappe_itodd.json` | `cd3300ce053ee425be4b8bd9c003bfd4d08f2b6dc2153496d1ce74d5c57900dd` |
| itodd | muse | `data/detections/muse/muse-full_itodd-test.json` | `2d34ebce3a464f129f6cdc8770686df56869eafa8bd8fff135fbfecb3c65813a` |
| hb | cnos | `data/detections/cnos/cnos-fastsam_hb-test.json` | `7eb39ad0d82783dc59a49cd2f6654c99b63d3b3ef3f051f3368056755e94e6b0` |
| hb | sam6d | `data/detections/sam6d/sam6d_official_hb.json` | `f22f496109341f8bb0f03c0d33476bb0af69f468f604afbd7fc03c898dc2d39a` |
| hb | nids | `data/detections/nids/nids_wa_sappe_hb.json` | `1bac5e38fc97a6810c43adb6b733daa7ba533358a7e1c49773d543aff7f7a0d9` |
| hb | muse | `data/detections/muse/muse-full_hb-test.json` | `c0e0802a3db1e2394507099098ed5000208d93e1701f8b19850d6cd6d7d59d1d` |
| ycbv | cnos | `data/detections/cnos/cnos-fastsam_ycbv-test.json` | `fdec15729676e15876302fc620f752cc5290ee28da5fc3c7e17da1072fd4f422` |
| ycbv | sam6d | `data/detections/sam6d/sam6d_official_ycbv.json` | `2288e24bfcbed29aedb719b53bff40f1a558a47f02392d5da0b6dabb2539abf8` |
| ycbv | nids | `data/detections/nids/nids_wa_sappe_ycbv.json` | `6eb751b20898e5cc8f499922590e9a07c2a645cfb7d5d14f7c59cb0d51c8544a` |
| ycbv | muse | `data/detections/muse/muse-full_ycbv-test.json` | `b4703a218d13f707d47556b2733eeddc38fea7d89bf927d113da25349c74f497` |

Preflight per dataset: layout/detection checksum match against this table ->
numeric smoke -> full run -> acceptance -> time-normalized private submission.
