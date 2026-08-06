# REPRODUCTION — parity ledger vs the `gedi` archive

The reproduction study lives in the frozen archive repo (`../gedi`, see its
`EXPERIMENTS.md` / `DISSERTATION_PLAN.md`). Before the dissertation cites
popoe-produced numbers, every headline result must be re-run through popoe
entrypoints and logged here. Until a row is checked, the gedi number remains
the authoritative (historical) figure.

**Acceptance rule**: same BOP test split, same recipe, full BOP AR within
±0.003 of the archive number (RANSAC stochasticity; tighten to bit-identical
if the run is seeded). Record the popoe commit hash for every reproduced
number.

## Context: the popoe promotion line already exists

popoe's own union2 + S_coarse campaign (2026-07-17 full-set run, 07-21 grasp
follow-up) already produced a popoe-native formal line: YCB-V full BOP AR
**0.8201**, LM-O **0.6896**, grasp ADD(-S)@0.1d 0.8616 / 0.7816. Artifacts:
`../gedi/ycbv_local_data/union_scoring_20260716/`. That line needs no parity
run — it was born on popoe. The ledger below is about re-producing the **gedi
script line** (the dissertation's reproduction headline) through popoe
entrypoints, under the dual-disclosure discipline of `../gedi/EXPERIMENTS.md`
§0: the reproduction headline is never rewritten by the popoe line.

## FreeZeV2 §IV-A experimental-setup conformance audit (2026-08-06)

Source: the Experimental Setup section (**§IV-A**, under "IV. Results") of
`2506.09784.pdf` = `papers/2506.09784v1.pdf` in the gedi archive. That single
tech report is the source for **both** FreeZeV2 (Table II Row 18) and
FreeZeV2-Accurate (Row 19); Row 19 is the BOP Challenge 2024 winner and is
identified by the paper itself as "entry **FreeZeV2.1** in the leaderboard"
(= `method_info/905`). The v2.1-only deltas are audited in a separate table
below, since §IV-A describes the shared setup, not Row 19's extras (those are
stated in §IV-D, Quantitative results).
Code inspected: the working tree based on `1d785b7` (dirty at audit time).
This is a setup/protocol audit, not a result row or a reproducible run identity;
re-check the findings against the eventual committed diff.

Status meanings: **match** = the paper setting and executed path agree;
**approximation** = an explicit local substitute; **partial** = the number is
present but its scope or semantics differs; **missing** = the paper protocol is
not implemented by the formal runner.

| Paper setting | Current popoe formal path | Status / disclosure |
|---|---|---|
| CNOS, SAM-6D, NIDS and MUSE, evaluated individually or as an ensemble (§IV-A verbatim; the ensemble rows 18/19 use all four) | `scripts/faithful_eval.sh` is CNOS-only. The A ensemble recipe below is three-source; the B four-source recipe is tuned rather than paper-faithful. Four-source detection assets are currently complete only for LM-O and YCB-V, not all seven BOP-Classic-Core datasets. | **Partial.** Do not describe either A/three-way or B/four-way as an exact reproduction of the complete segmentation setup. **Why A is three-way**: a **project scoping decision, not an availability limit** — official MUSE masks for LM-O and YCB-V were already in `data/detections/muse/` when the recipe froze on 2026-07-30, and the other five core sets are downloadable too (see the resolved MUSE note below). Never justify three-way as a MUSE-availability limit. Do *not* justify three-way as "matching `method_info/756`'s three-source self-description" — three-way appears nowhere in the paper, and 756 is the only FreeZeV2 config with no corresponding paper row. |
| 162 templates per object, using the CNOS camera viewpoints | Faithful pins set `POPOE_N_VIEWS=162` and `POPOE_QUERY_VIEWS=ico162`. | **Match.** |
| Retain raw query points visible in at least `V=18` views | Faithful pins set `POPOE_QUERY_MIN_VIEWS=18`; filtering is implemented in `src/popoe/freeze/feature_extractor.py`. | **Match.** |
| Render at 480×480 with the object occupying approximately 50% of width/height | Faithful pins use `POPOE_QUERY_CANON=476` and `POPOE_QUERY_FILL=0.5`; 476 is the local DINO patch-grid-compatible substitute. | **Approximation.** Always disclose 476/0.5, not “exact 480×480”. |
| ViT-giant DINOv2 patch features from intermediate layers, following FoundPose | The backbone is `dinov2_vitg14_reg`. The implementation selects one inferred FoundPose-style layer (block 30 for ViT-g), because the public text does not pin an exact block/list. | **Partial / pinned-by-us.** Backbone matches; layer selection is a local inference. |
| Query point set 5k | The adapter samples 5k surface points before the `V=18` visibility gate; the final retained query set can therefore contain fewer than 5k points. | **Partial.** The paper's stated 5k budget is not enforced after filtering. |
| Dense target point set 3k | `--icp-dense --icp-dense-max 3000` caps the separately reconstructed ICP cloud. GeDi target encoding still builds neighborhoods from all valid mask-depth pixels. | **Missing as a shared budget.** The 3k cap applies to ICP only, not to the dense target set used throughout feature extraction. |
| Sparse target point set at most 256 | Faithful recipes use `--grid 16`, giving at most 256 samples. The implementation uses a rectangular mask grid and target-side bilinear feature sampling rather than the paper-highlighted square/no-interpolation behavior. | **Partial.** The count matches; sampling semantics do not. |
| Top-`k=10` feature correspondences | Faithful recipes pass `--corr-topk 10`. | **Match for the configured solver path.** |
| Localization keeps `M=N+1` proposals | `examples/bop_eval.py` floors `M` using the dataset-wide maximum target multiplicity. This gives `M=2` on the single-instance LM-O/YCB-V cases, but does not compute `N+1` per target on general multi-instance datasets. | **Partial / missing generally.** |
| Detection uses `M=100` and discards masks below `tau_mask=0.4` | The formal runner consumes BOP `test_targets_bop19.json` and exposes neither this detection-mode proposal count nor the paper confidence cutoff. | **Missing.** Current formal runs are localization-protocol runs. |
| Two-scale GeDi: 32D per scale, neighborhoods at 30% and 40% of object diameter | `_TwoScaleGeDi` concatenates two 32D descriptors; faithful diameter normalization makes radii 0.3 and 0.4 object diameter. | **Match.** |
| PCA/fusion output dimension 128 | Two-scale geometry is 64D; visual PCA defaults to 64D; normalized concatenation produces 128D. | **Match.** |
| RANSAC inlier and ICP thresholds are 3% of object diameter | Faithful recipes pass `--tau-diameter`; the threshold is derived from the diameter-normalized canonical frame. | **Match.** |
| Parallel GPU RANSAC, 10,000 iterations, selected with the feature-aware score | The iteration count is 10,000, but formal faithful recipes select `--solver o3d`. That path is CPU Open3D geometry RANSAC and does not execute the paper's Eq. 5 feature-aware hypothesis selection. `GPURansacSolver` contains the feature-aware score but is not wired into these recipes. | **Missing on the executed path; critical deviation.** |
| Timing hardware: NVIDIA A40 and Xeon Silver 4316 @ 2.30 GHz | Recorded project runs use the lab 4×RTX 4090 host or other stated infrastructure; recipes do not assert the paper CPU/GPU model. | **Hardware mismatch.** Accuracy results may still be compared with full disclosure; runtime/FPS must not be presented as paper-hardware parity. |

Additional algorithmic variable: every formal recipe below enables
`--render-rerank`, which is not specified in §IV-A. It must be reported as a
popoe extension rather than folded into the “faithful” label.

### v2.1-only deltas (Table II Row 19 = FreeZeV2-Accurate = `method_info/905`)

Row 19 is the strongest configuration **in Table II of this report**, and has
full public per-set scores on the BOP leaderboard (LM-O 0.771 / YCB-V 0.915;
§IV-D reports mean 82.1 AR, and gedi `BOP_OFFICIAL_BASELINES.md` note 1 records
that 905 matches Row 19 digit-for-digit across all seven sets).

**It is not the strongest published FreeZe config, and not the leaderboard
top.** `method_info/1063` = **FreeZeV2.2** (2025-05-31) scores ARCore **0.833**
vs v2.1's 0.821 — LM-O 0.777 / YCB-V 0.918 (gedi
`BOP_OFFICIAL_BASELINES.md`). Its stated delta is feature-similarity in the
RANSAC fitness, and its **segmentor composition is publicly undeclared**, which
is why gedi `DISSERTATION_PLAN.md` treats v2.2 as a Ch2 frontier reference point
only and never as a like-for-like target. Above v2.2 the board itself has
FRTPose-WAPR.v2 at 0.837. So qualify the superlative every time: Row 19 is the
best config *in this paper*, v2.1 is the best *documented and decomposable
ceiling* for this audit, and neither is state of the art.

Row 19 is nonetheless the right **ceiling** reference for this audit, because it
is the strongest config whose recipe is documented well enough to enumerate
deltas at all. Each delta has a different kind of unavailability, and
they must not be collapsed into one "missing":

| v2.1 delta | Status | Reason / what it would take |
|---|---|---|
| `M = 2N` masks per segmentation model | **Missing, but alignable.** No closed dependency. | **Two distinct sources, do not conflate.** (a) *Where `2N` is stated*: §IV-D prose on Row 19 only — "increases the number of processed masks up to `M = 2N`". No ablation, no per-set numbers, no timing for `2N` anywhere in the report. (b) *What Table V actually ablates*: `M ∈ {N, N+1, N+2}` for the **base FreeZeV2** localization protocol (73.7 / 75.4 / 75.6 mean AR at 1.2 / 1.5 / 1.7 s), establishing that `N+1` is the paper's default and that returns are already flattening by `N+2` (+0.2). Table V therefore **does not validate `2N`** — it neither measures it nor bounds it; the `N+2` trend is only weak evidence that `2N`'s gain is small and its cost is not. What makes `2N` alignable is that the *quantity* `M` is public and parameter-free, not that it was ablated. Prerequisite is the per-target `N+1` fix already listed above — `examples/bop_eval.py` currently floors `M` from the dataset-wide maximum multiplicity, giving `M=2` on single-instance LM-O/YCB-V, so there is no per-target `N` to double yet. Once `N` is per-target, `2N` is a coefficient change; report any `2N` run as our own measurement, since the paper gives no `2N` number to compare against. |
| Symmetry-Aware Refinement (SAR) | **Approximation ceiling: implementable, not verifiable as equivalent.** | §IV-D only says v2.1 "integrates Symmetry-Aware Refinement (SAR) [9]" — ref [9] is FreeZe v1 (`2312.00947v3` §3.6, "based on rendering and visual features"). So the spec lives in a *different* paper and there is no public code. popoe's `--render-rerank` reorders PCA flip variants only (see every recipe's Cautions row). gedi `scripts/freezev2_sym_refine.py` is closer (PCA 3 axes × 36 angles, Chamfer < 1% diameter, ≤32 symmetries) but is our own symmetry enumeration, not a port of v1 SAR. Any implementation must be disclosed as an approximation of SAR, never as SAR. |
| Improved scoring by comparing visual features of input image vs rendered pose | **Missing.** | §IV-D gives one prose sentence, no equation and no parameters. Not reconstructible to a verifiable spec from the public text. |

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
`muse-repro` is *not* required to cover them. Two claims are withdrawn:
"no downloadable masks" (gedi `notes.md`) and "authors only published these
two sets / 不可能" (gedi `TODO.md`) — both were wrong.

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
therefore keeps its original role — evidence for the T1 from-paper reproduction
claim in gedi `EXPERIMENTS.md`, not a replacement for official masks in pose
runs.

Consequence for the dissertation: the MUSE artefacts and the `M=2N` rule are
obtainable, but exact Row 19 recipe parity remains unverifiable because SAR and
render scoring are underspecified. The distance to 905 is confounded by SAR +
`M=2N` + render scoring — report it as a not-yet-matched ceiling, never as a
reproduction residual. The
detection-matched residual stays at A/three-way vs `method_info/756`, and even
that is matched only in **detection source count**: the critical-deviation row
above (`--solver o3d` bypassing the Eq. 5 feature-aware selector) plus
`--render-rerank` mean the *pose stage* is not recipe-matched in either
direction. Prose must say "detection-matched", never "recipe-matched".

Required follow-up before claiming exact setup parity:

- [ ] Route the paper-faithful recipe through 10,000-iteration GPU
  feature-aware RANSAC and verify that Eq. 5 is the executed selector.
- [ ] Define one 3k dense target cloud and reuse it consistently for GeDi and
  ICP, or document and ablate the split budgets.
- [ ] Implement per-target `M=N+1` and a separate detection protocol with
  `M=100`, `tau_mask=0.4`. This is a prerequisite for the v2.1 `M=2N` row
  above, and also touches the multi-instance sets (ITODD / IC-BIN / T-LESS)
  whose per-dataset spread is still recorded as unexplained below.
- [ ] Resolve or explicitly preserve the 480→476, post-visibility 5k, and
  sparse-target sampling approximations.
- [ ] Add a configuration/contract test covering the complete paper setup.

## Two-line formal recipes (frozen 2026-07-30)

> Formal score = BOP evaluation server only. Local full AR in
> `AR_SUMMARY.md` is a development self-check, not the dissertation score.
> All four recipes are pinned to code tag **`twoline-rerank-fix-20260731`**
> (commit pin recorded in each recipe table; verify with
> `git rev-parse HEAD` == `git rev-parse twoline-rerank-fix-20260731^{commit}`,
> **not** tag-name-only `describe` — this annotated tag was moved once already).
> Run root default: `/workspace/results/twoline_20260731_rerankfix`
> (void batch lived at `…/twoline_20260730` — do not write new poses there).
> Orchestration / go-no-go: `../gedi/EXPERIMENT_PLAN.md`.
>
> > ⚠️ **The 2026-07-30 batch of eight runs is void.** It ran at
> > `twoline-prep-20260730a`, where `--render-rerank` re-ICP'd only the flipped
> > variants and did so at a 4-10x too loose threshold, inflating their `s_icp`
> > 3-4x and corrupting selection (LM-O AR(2/3) 0.2489, vs 0.7745 for the same
> > candidate pool with those picks excluded). Fixed in PR #29; this tag is the
> > first one whose runs count. **src/ changed, so the run identity changed** —
> > do not quote any number produced under the old tag. The ~10GB of feature
> > caches on volume `8rf4r42sf1` stay valid (PR #29 touches only the
> > pose/scoring path; the stage-cache key is byte-identical), but every
> > `poses.csv` / `cand.csv` from that batch must be moved aside before a
> > re-run — `bop_eval` resumes by ROW COUNT (`adapters.resolve_resume`), so a
> > stale CSV makes the re-run declare every target done and exit "successfully"
> > with the bad data still in place. **Do not BOP-submit any void-batch CSV.**
>
> Values marked
> **pinned-by-us** are frozen project choices where the public recipe is silent:
> `--seed 42` (re-pinned from 1234 on 2026-07-30, Vincent's call — conventional
> value; the smoke batch ran at 1234, which validated mechanics only and is
> unaffected), A-line Eq.7 unit exponents (`alpha=beta=gamma=1`) for the
> `--use-s-coarse` product term, and `POPOE_QUERY_CANON=476` (render canvas;
> the paper names 480²/50% — 476/0.5 is our measured-equivalent pin). The
> `--render-rerank` switch is score-affecting, so every one of the eight runs
> below must use fresh `poses.csv` and `cand.csv` paths.
>
> **Smoke first**: before any full run, execute the same block with
> `--objs 1` appended and `smoke_`-prefixed `--out/--cand-csv` paths
> (cache may point at a verified warm directory; see gedi experiment plan §3.4).
> Reference smoke: **`tuned-4way` LM-O `--objs 1`** (~175 targets). Every
> recipe needs its own S1/S2/S3 record before that recipe's full run.
>
> > **A smoke must assert a NUMBER, not just survival.** The 2026-07-29
> > RERANK-SMOKE passed on "every env pin echoed, rerank breakdown lines
> > present, zero Tracebacks" — and the stage it was gating was, at that moment,
> > destroying 52 AR points. Liveness criteria cannot see a wrong answer. Any
> > stage that can change the score must clear a falsifiable numeric bar.
> > **S1 and S2 are co-equal hard gates; S3 alone is never enough.**
> >
> > 1. **Measurement symmetry (S1, hard, one-sided)** — run the shipped checker,
> >    do not re-implement ad hoc:
> >    ```bash
> >    python scripts/check_rerank_symmetry.py smoke_cand.csv   # exit 0 = pass
> >    ```
> >    **Fail only if** flipped/unflipped `s_icp` median ratio **> 1.4**
> >    (flip *inflation* — the 2026-07-30 bug was 3.0-4.1x). After PR #29 every
> >    variant is re-ICP'd at the real tau, so wrong flips get *lower* fitness;
> >    a ratio **below 0.7 prints WARN but still passes** (smoke 2026-07-31:
> >    0.17x with AR(2/3) 0.816). Bilateral `[0.7, 1.4]` is retired — it
> >    conflated "different measurement" with "different pose quality".
> > 2. **Score floor (S2, hard)** — `ar_flat` on the smoke poses CSV.
> >    Reference smoke (`tuned-4way` LM-O `--objs 1`): **AR(2/3) ≥ 0.65**.
> >    Offline on the void-batch cand for that subset: bad-code ~**0.036**,
> >    exclude-flipped reselect (rule C ≈ rerank-off) ~**0.816**. Use 0.65 as
> >    a floor with margin — **do not treat 0.816 as a post-fix must-hit**
> >    (that is rerank-off waterline, not measured post-fix AR).
> >    Optional one-sided relative check: `rerank-on ≥ exclude_flipped − 5pt`
> >    (no upper cap; a large gain must not FAIL).
> >    **Forbidden reference:** reselect using pre-rerank pose of the *chosen*
> >    hyp (triage rule B; measured worse than the bug). Exclude *flipped*
> >    candidates, then argmax — that is the only valid control pool.
> > 3. **Liveness (S3)** — env pins echoed, rerank breakdown lines, 0
> >    Tracebacks, rows > 0. Necessary, never sufficient.
>
> **Post-run (no GPU)**: per dataset dir fill the remaining four artifacts —
> `RECIPE.md` (copy the exact block + commit + date), `AR_SUMMARY.md`
> (local `ar_flat`/VSD scripts; self-check only), `bop_server.md` (score +
> submission id after the private upload), `grasp_summary.md` (same-CSV
> ADD(-S) via the gedi grasp script).

Artifact convention for every dataset run follows `../gedi/CONSOLIDATION.md`
§3.2:

```
out_dir/
  poses.csv        # main pose CSV
  cand.csv         # candidate-level dump (--cand-csv; replay/ablation input)
  RECIPE.md        # flags, detection sources, commit, seed, date
  AR_SUMMARY.md    # local full AR + MSSD/MSPD/VSD self-check only
  bop_server.md    # official BOP server score + submission id
  grasp_summary.md # ADD(-S) computed from the same poses.csv
```

### faithful-cnos

| Field | Frozen value |
|---|---|
| Report point | A / single-source |
| Code | `twoline-rerank-fix-20260731` (`git rev-parse twoline-rerank-fix-20260731^{commit}` = `509072e`; includes one-sided S1 smoke checker) |
| Datasets | LM-O + YCB-V; one full BOP test run each |
| Detection inputs | CNOS only: `data/detections/cnos/cnos-fastsam_lmo-test.json`, `data/detections/cnos/cnos-fastsam_ycbv-test.json` |
| Scoring | Paper Eq.7 three-term form with `--use-s-coarse`; Eq.7 exponents are **pinned-by-us** to unit exponents |
| Leaderboard comparator | A / single-source -> FreeZe(CNOS) LM-O/YCB-V = 0.689 / 0.853 |
| Artifacts | `$RUN/{lmo,ycbv}/` each contains `poses.csv`, `cand.csv`, `RECIPE.md`, `AR_SUMMARY.md`, `bop_server.md`, `grasp_summary.md` |
| Cautions | Rerank scope differs from official SAR: popoe only reorders PCA flip variants. Dense resampling uses `rng(0)` where invoked and is independent of `--seed`. Encoding degradation is explicit in logs as `DEGRADE`. These runs are not bit/row comparable to historical anchors because seed, rerank and implementation fixes are new variables. |

```bash
set -euo pipefail

POPOE="${POPOE:-/workspace/popoe}"
BOP="${BOP:-/workspace/bop_data}"
DET="${DET:-$POPOE/data/detections}"
RUN_ROOT="${RUN_ROOT:-/workspace/results/twoline_20260731_rerankfix}"
RUN="$RUN_ROOT/faithful-cnos"
PY="${PY:-python}"
SEED=42

cd "$POPOE"
# Pin by dereferenced tag commit, not tag *name* via describe. This annotated
# tag was moved (22653d2 → 2b8c0eb → 509072e); plain `git fetch --tags` does
# not clobber a local tag, so a stale tip can still `describe` clean while
# missing scripts/check_rerank_symmetry.py. On pods: fresh clone +
# `git fetch --tags --force` before checkout. Optional POPOE_PIN=fullsha
# forces an exact match (set it from the recipe Code row after retag).
TAG=twoline-rerank-fix-20260731
TAG_COMMIT=$(git rev-parse "${TAG}^{commit}" 2>/dev/null) || {
  echo "tag $TAG missing; run: git fetch --tags --force" >&2
  exit 1
}
if [ "$(git rev-parse HEAD)" != "$TAG_COMMIT" ]; then
  echo "wrong popoe checkout; need $TAG @ $TAG_COMMIT (got HEAD=$(git rev-parse HEAD))" >&2
  exit 1
fi
if [ -n "${POPOE_PIN:-}" ] && [ "$(git rev-parse HEAD)" != "$POPOE_PIN" ]; then
  echo "POPOE_PIN=$POPOE_PIN does not match HEAD=$(git rev-parse HEAD)" >&2
  exit 1
fi
if [ ! -f scripts/check_rerank_symmetry.py ]; then
  echo "missing scripts/check_rerank_symmetry.py — stale tag tip; fetch --tags --force" >&2
  exit 1
fi
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
export POPOE_QUERY_MIN_VIEWS=18
export POPOE_QUERY_VIEWS=ico162

mkdir -p "$RUN/lmo" "$RUN/ycbv"
for f in "$RUN/lmo/poses.csv" "$RUN/lmo/cand.csv" \
         "$RUN/ycbv/poses.csv" "$RUN/ycbv/cand.csv"; do
  # NOT `test ! -e a && test ! -e b`: under set -e a failing left-hand test
  # is errexit-exempt and the guard silently passes (second-review P0).
  if [ -e "$f" ]; then echo "refusing: $f exists — rerank requires FRESH --out" >&2; exit 1; fi
done

"$PY" examples/bop_eval.py \
  --bop "$BOP/lmo" --dataset lmo \
  --detections "$DET/cnos/cnos-fastsam_lmo-test.json" \
  --merge none --topk 2 --grid 16 --solver o3d --seed "$SEED" \
  --weights 1.0 \
  --use-s-coarse \
  --corr-topk 10 \
  --tau-diameter \
  --icp-dense --icp-dense-max 3000 \
  --render-rerank \
  --render-backend nvdiffrast \
  --out "$RUN/lmo/poses.csv" --cache "$RUN/lmo/cache" \
  --cand-csv "$RUN/lmo/cand.csv"

"$PY" examples/bop_eval.py \
  --bop "$BOP/ycbv" --dataset ycbv \
  --detections "$DET/cnos/cnos-fastsam_ycbv-test.json" \
  --merge none --topk 2 --grid 16 --solver o3d --seed "$SEED" \
  --weights 1.0 \
  --use-s-coarse \
  --corr-topk 10 \
  --tau-diameter \
  --icp-dense --icp-dense-max 3000 \
  --render-rerank \
  --render-backend nvdiffrast \
  --out "$RUN/ycbv/poses.csv" --cache "$RUN/ycbv/cache" \
  --cand-csv "$RUN/ycbv/cand.csv"
```

### faithful-3way

| Field | Frozen value |
|---|---|
| Report point | A / three-way |
| Code | `twoline-rerank-fix-20260731` (`git rev-parse twoline-rerank-fix-20260731^{commit}` = `509072e`; includes one-sided S1 smoke checker) |
| Datasets | LM-O + YCB-V; one full BOP test run each |
| Detection inputs | CNOS + SAM6D + NIDS official JSONs under `data/detections/`; `--merge none` keeps the paper-style union unfiltered |
| Scoring | Paper Eq.7 three-term form with `--use-s-coarse`; Eq.7 exponents are **pinned-by-us** to unit exponents |
| Leaderboard comparator | A / three-way -> FreeZeV2(756) LM-O/YCB-V = 0.764 / 0.906 — **detection-matched** to 756's three-source self-description (no MUSE) |
| Artifacts | `$RUN/{lmo,ycbv}/` each contains `poses.csv`, `cand.csv`, `RECIPE.md`, `AR_SUMMARY.md`, `bop_server.md`, `grasp_summary.md` |
| Cautions | Rerank scope differs from official SAR: popoe only reorders PCA flip variants. Dense resampling uses `rng(0)` where invoked and is independent of `--seed`. Encoding degradation is explicit in logs as `DEGRADE`. These runs are not bit/row comparable to historical anchors because seed, rerank and implementation fixes are new variables. |

```bash
set -euo pipefail

POPOE="${POPOE:-/workspace/popoe}"
BOP="${BOP:-/workspace/bop_data}"
DET="${DET:-$POPOE/data/detections}"
RUN_ROOT="${RUN_ROOT:-/workspace/results/twoline_20260731_rerankfix}"
RUN="$RUN_ROOT/faithful-3way"
PY="${PY:-python}"
SEED=42

cd "$POPOE"
# Pin by dereferenced tag commit, not tag *name* via describe. This annotated
# tag was moved (22653d2 → 2b8c0eb → 509072e); plain `git fetch --tags` does
# not clobber a local tag, so a stale tip can still `describe` clean while
# missing scripts/check_rerank_symmetry.py. On pods: fresh clone +
# `git fetch --tags --force` before checkout. Optional POPOE_PIN=fullsha
# forces an exact match (set it from the recipe Code row after retag).
TAG=twoline-rerank-fix-20260731
TAG_COMMIT=$(git rev-parse "${TAG}^{commit}" 2>/dev/null) || {
  echo "tag $TAG missing; run: git fetch --tags --force" >&2
  exit 1
}
if [ "$(git rev-parse HEAD)" != "$TAG_COMMIT" ]; then
  echo "wrong popoe checkout; need $TAG @ $TAG_COMMIT (got HEAD=$(git rev-parse HEAD))" >&2
  exit 1
fi
if [ -n "${POPOE_PIN:-}" ] && [ "$(git rev-parse HEAD)" != "$POPOE_PIN" ]; then
  echo "POPOE_PIN=$POPOE_PIN does not match HEAD=$(git rev-parse HEAD)" >&2
  exit 1
fi
if [ ! -f scripts/check_rerank_symmetry.py ]; then
  echo "missing scripts/check_rerank_symmetry.py — stale tag tip; fetch --tags --force" >&2
  exit 1
fi
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
export POPOE_QUERY_MIN_VIEWS=18
export POPOE_QUERY_VIEWS=ico162

mkdir -p "$RUN/lmo" "$RUN/ycbv"
for f in "$RUN/lmo/poses.csv" "$RUN/lmo/cand.csv" \
         "$RUN/ycbv/poses.csv" "$RUN/ycbv/cand.csv"; do
  # NOT `test ! -e a && test ! -e b`: under set -e a failing left-hand test
  # is errexit-exempt and the guard silently passes (second-review P0).
  if [ -e "$f" ]; then echo "refusing: $f exists — rerank requires FRESH --out" >&2; exit 1; fi
done

"$PY" examples/bop_eval.py \
  --bop "$BOP/lmo" --dataset lmo \
  --sources "cnos=$DET/cnos/cnos-fastsam_lmo-test.json,sam6d=$DET/sam6d/sam6d_ism_lmo.json,nids=$DET/nids/nids_wa_sappe_lmo.json" \
  --merge none --topk 2 --grid 16 --solver o3d --seed "$SEED" \
  --weights 1.0 \
  --use-s-coarse \
  --corr-topk 10 \
  --tau-diameter \
  --icp-dense --icp-dense-max 3000 \
  --render-rerank \
  --render-backend nvdiffrast \
  --out "$RUN/lmo/poses.csv" --cache "$RUN/lmo/cache" \
  --cand-csv "$RUN/lmo/cand.csv"

"$PY" examples/bop_eval.py \
  --bop "$BOP/ycbv" --dataset ycbv \
  --sources "cnos=$DET/cnos/cnos-fastsam_ycbv-test.json,sam6d=$DET/sam6d/sam6d_ism_ycbv.json,nids=$DET/nids/nids_wa_sappe_ycbv.json" \
  --merge none --topk 2 --grid 16 --solver o3d --seed "$SEED" \
  --weights 1.0 \
  --use-s-coarse \
  --corr-topk 10 \
  --tau-diameter \
  --icp-dense --icp-dense-max 3000 \
  --render-rerank \
  --render-backend nvdiffrast \
  --out "$RUN/ycbv/poses.csv" --cache "$RUN/ycbv/cache" \
  --cand-csv "$RUN/ycbv/cand.csv"
```

### tuned-cnos

| Field | Frozen value |
|---|---|
| Report point | B / single-source |
| Code | `twoline-rerank-fix-20260731` (`git rev-parse twoline-rerank-fix-20260731^{commit}` = `509072e`; includes one-sided S1 smoke checker) |
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
RUN_ROOT="${RUN_ROOT:-/workspace/results/twoline_20260731_rerankfix}"
RUN="$RUN_ROOT/tuned-cnos"
PY="${PY:-python}"
SEED=42

cd "$POPOE"
# Pin by dereferenced tag commit, not tag *name* via describe. This annotated
# tag was moved (22653d2 → 2b8c0eb → 509072e); plain `git fetch --tags` does
# not clobber a local tag, so a stale tip can still `describe` clean while
# missing scripts/check_rerank_symmetry.py. On pods: fresh clone +
# `git fetch --tags --force` before checkout. Optional POPOE_PIN=fullsha
# forces an exact match (set it from the recipe Code row after retag).
TAG=twoline-rerank-fix-20260731
TAG_COMMIT=$(git rev-parse "${TAG}^{commit}" 2>/dev/null) || {
  echo "tag $TAG missing; run: git fetch --tags --force" >&2
  exit 1
}
if [ "$(git rev-parse HEAD)" != "$TAG_COMMIT" ]; then
  echo "wrong popoe checkout; need $TAG @ $TAG_COMMIT (got HEAD=$(git rev-parse HEAD))" >&2
  exit 1
fi
if [ -n "${POPOE_PIN:-}" ] && [ "$(git rev-parse HEAD)" != "$POPOE_PIN" ]; then
  echo "POPOE_PIN=$POPOE_PIN does not match HEAD=$(git rev-parse HEAD)" >&2
  exit 1
fi
if [ ! -f scripts/check_rerank_symmetry.py ]; then
  echo "missing scripts/check_rerank_symmetry.py — stale tag tip; fetch --tags --force" >&2
  exit 1
fi
if [ -n "$(git status --porcelain)" ]; then
  echo "refusing dirty popoe worktree" >&2
  exit 1
fi

export BOP DET RUN SEED
export OMP_NUM_THREADS=16
export TORCH_HOME="${TORCH_HOME:-/workspace/torch_cache}"
export POPOE_GEDI_PATH=/workspace/gedi
export POPOE_BOP_TOOLKIT=/workspace/bop_toolkit
unset POPOE_CANON_BASIS POPOE_QUERY_POINTS POPOE_N_VIEWS POPOE_QUERY_CANON
unset POPOE_QUERY_FILL POPOE_QUERY_MIN_VIEWS POPOE_QUERY_VIEWS
unset POPOE_TARGET_GRID POPOE_TARGET_CANON POPOE_TARGET_FILL POPOE_TARGET_CROP
unset POPOE_VIS_DIM POPOE_VIS_WEIGHT POPOE_SKIP_VIS POPOE_DINO_LAYER
unset POPOE_TWO_SCALE_GEDI POPOE_DGEDI_MODE POPOE_GEOM_BACKBONE POPOE_MESH_SHADING
unset POPOE_FPFH_RADII POPOE_FPFH_VOXEL_FRAC POPOE_FPFH_NORMAL_FRAC POPOE_FPFH_ORIENT

mkdir -p "$RUN/lmo" "$RUN/ycbv"
for f in "$RUN/lmo/poses.csv" "$RUN/lmo/cand.csv" \
         "$RUN/ycbv/poses.csv" "$RUN/ycbv/cand.csv"; do
  # NOT `test ! -e a && test ! -e b`: under set -e a failing left-hand test
  # is errexit-exempt and the guard silently passes (second-review P0).
  if [ -e "$f" ]; then echo "refusing: $f exists — rerank requires FRESH --out" >&2; exit 1; fi
done

"$PY" examples/bop_eval.py \
  --bop "$BOP/lmo" --dataset lmo \
  --detections "$DET/cnos/cnos-fastsam_lmo-test.json" \
  --merge none --topk 2 --grid 32 --solver o3d --seed "$SEED" \
  --weights 1.0,0.7,0.5,0.3,0.2 \
  --render-rerank \
  --render-backend nvdiffrast \
  --out "$RUN/lmo/poses.csv" --cache "$RUN/lmo/cache" \
  --cand-csv "$RUN/lmo/cand.csv"

"$PY" examples/bop_eval.py \
  --bop "$BOP/ycbv" --dataset ycbv \
  --detections "$DET/cnos/cnos-fastsam_ycbv-test.json" \
  --merge ycbv --topk 2 --grid 32 --solver o3d --seed "$SEED" \
  --weights 1.0,0.7,0.5,0.3,0.2 \
  --use-s-coarse \
  --render-rerank \
  --render-backend nvdiffrast \
  --out "$RUN/ycbv/poses.csv" --cache "$RUN/ycbv/cache" \
  --cand-csv "$RUN/ycbv/cand.csv"
```

### tuned-4way

| Field | Frozen value |
|---|---|
| Report point | B / four-way |
| Code | `twoline-rerank-fix-20260731` (`git rev-parse twoline-rerank-fix-20260731^{commit}` = `509072e`; includes one-sided S1 smoke checker) |
| Datasets | LM-O + YCB-V; one full BOP test run each |
| Detection inputs | CNOS + SAM6D + NIDS + official MUSE JSONs under `data/detections/`; `muse` means downloaded official artefacts, not `muse-repro` |
| Scoring | Campaign2 tuned ChampionScorer: grid32, weights `1.0,0.7,0.5,0.3,0.2`; YCB-V uses `--merge ycbv --use-s-coarse`, LM-O uses `--merge none` and no `--use-s-coarse` |
| Leaderboard comparator | B / four-way -> FreeZeV2(756) LM-O/YCB-V = 0.764 / 0.906 — detection carries **one extra source (MUSE)** vs 756's self-description; the detection-matched like-for-like lives at A/three-way, state this in prose |
| Artifacts | `$RUN/{lmo,ycbv}/` each contains `poses.csv`, `cand.csv`, `RECIPE.md`, `AR_SUMMARY.md`, `bop_server.md`, `grasp_summary.md` |
| Cautions | Rerank scope differs from official SAR: popoe only reorders PCA flip variants. Dense resampling uses `rng(0)` where invoked and is independent of `--seed`. Encoding degradation is explicit in logs as `DEGRADE`. These runs are not bit/row comparable to historical anchors because seed, rerank and implementation fixes are new variables. |

```bash
set -euo pipefail

POPOE="${POPOE:-/workspace/popoe}"
BOP="${BOP:-/workspace/bop_data}"
DET="${DET:-$POPOE/data/detections}"
RUN_ROOT="${RUN_ROOT:-/workspace/results/twoline_20260731_rerankfix}"
RUN="$RUN_ROOT/tuned-4way"
PY="${PY:-python}"
SEED=42

cd "$POPOE"
# Pin by dereferenced tag commit, not tag *name* via describe. This annotated
# tag was moved (22653d2 → 2b8c0eb → 509072e); plain `git fetch --tags` does
# not clobber a local tag, so a stale tip can still `describe` clean while
# missing scripts/check_rerank_symmetry.py. On pods: fresh clone +
# `git fetch --tags --force` before checkout. Optional POPOE_PIN=fullsha
# forces an exact match (set it from the recipe Code row after retag).
TAG=twoline-rerank-fix-20260731
TAG_COMMIT=$(git rev-parse "${TAG}^{commit}" 2>/dev/null) || {
  echo "tag $TAG missing; run: git fetch --tags --force" >&2
  exit 1
}
if [ "$(git rev-parse HEAD)" != "$TAG_COMMIT" ]; then
  echo "wrong popoe checkout; need $TAG @ $TAG_COMMIT (got HEAD=$(git rev-parse HEAD))" >&2
  exit 1
fi
if [ -n "${POPOE_PIN:-}" ] && [ "$(git rev-parse HEAD)" != "$POPOE_PIN" ]; then
  echo "POPOE_PIN=$POPOE_PIN does not match HEAD=$(git rev-parse HEAD)" >&2
  exit 1
fi
if [ ! -f scripts/check_rerank_symmetry.py ]; then
  echo "missing scripts/check_rerank_symmetry.py — stale tag tip; fetch --tags --force" >&2
  exit 1
fi
if [ -n "$(git status --porcelain)" ]; then
  echo "refusing dirty popoe worktree" >&2
  exit 1
fi

export BOP DET RUN SEED
export OMP_NUM_THREADS=16
export TORCH_HOME="${TORCH_HOME:-/workspace/torch_cache}"
export POPOE_GEDI_PATH=/workspace/gedi
export POPOE_BOP_TOOLKIT=/workspace/bop_toolkit
unset POPOE_CANON_BASIS POPOE_QUERY_POINTS POPOE_N_VIEWS POPOE_QUERY_CANON
unset POPOE_QUERY_FILL POPOE_QUERY_MIN_VIEWS POPOE_QUERY_VIEWS
unset POPOE_TARGET_GRID POPOE_TARGET_CANON POPOE_TARGET_FILL POPOE_TARGET_CROP
unset POPOE_VIS_DIM POPOE_VIS_WEIGHT POPOE_SKIP_VIS POPOE_DINO_LAYER
unset POPOE_TWO_SCALE_GEDI POPOE_DGEDI_MODE POPOE_GEOM_BACKBONE POPOE_MESH_SHADING
unset POPOE_FPFH_RADII POPOE_FPFH_VOXEL_FRAC POPOE_FPFH_NORMAL_FRAC POPOE_FPFH_ORIENT

mkdir -p "$RUN/lmo" "$RUN/ycbv"
for f in "$RUN/lmo/poses.csv" "$RUN/lmo/cand.csv" \
         "$RUN/ycbv/poses.csv" "$RUN/ycbv/cand.csv"; do
  # NOT `test ! -e a && test ! -e b`: under set -e a failing left-hand test
  # is errexit-exempt and the guard silently passes (second-review P0).
  if [ -e "$f" ]; then echo "refusing: $f exists — rerank requires FRESH --out" >&2; exit 1; fi
done

"$PY" examples/bop_eval.py \
  --bop "$BOP/lmo" --dataset lmo \
  --sources "cnos=$DET/cnos/cnos-fastsam_lmo-test.json,sam6d=$DET/sam6d/sam6d_ism_lmo.json,nids=$DET/nids/nids_wa_sappe_lmo.json,muse=$DET/muse/muse-full_lmo-test.json" \
  --merge none --topk 2 --grid 32 --solver o3d --seed "$SEED" \
  --weights 1.0,0.7,0.5,0.3,0.2 \
  --render-rerank \
  --render-backend nvdiffrast \
  --out "$RUN/lmo/poses.csv" --cache "$RUN/lmo/cache" \
  --cand-csv "$RUN/lmo/cand.csv"

"$PY" examples/bop_eval.py \
  --bop "$BOP/ycbv" --dataset ycbv \
  --sources "cnos=$DET/cnos/cnos-fastsam_ycbv-test.json,sam6d=$DET/sam6d/sam6d_ism_ycbv.json,nids=$DET/nids/nids_wa_sappe_ycbv.json,muse=$DET/muse/muse-full_ycbv-test.json" \
  --merge ycbv --topk 2 --grid 32 --solver o3d --seed "$SEED" \
  --weights 1.0,0.7,0.5,0.3,0.2 \
  --use-s-coarse \
  --render-rerank \
  --render-backend nvdiffrast \
  --out "$RUN/ycbv/poses.csv" --cache "$RUN/ycbv/cache" \
  --cand-csv "$RUN/ycbv/cand.csv"
```

## Headline ledger

> **2026-07-26 pipeline verify COMPLETE** — popoe **`75553a1`**, pod `wrmy8k0thtxjq6` (stopped).  
> Artifacts: `outputs/pipeline_verify_20260726/`. Full AR = mean(MSSD, MSPD, VSD).  
> **Δ exceeds ±0.003** on full AR; all six rows land **above** the gedi archive. Do **not** rewrite the archive headline; cite this as the popoe parity measurement (dual-track). Promotion line remains 0.8201 / 0.6896.
>
> **Calibre note (2026-07-29, post PR #23)**: every locally scored AR in this ledger is **legacy per-object calibre** (the pre-#23 scorer averaged objects with equal weight). BOP flat (per-instance) calibre, recomputed from the same CSVs: **#1 = 0.7892** (vs 0.7781), **#2 = 0.6844** (vs 0.6792); the gedi reference figures in flat calibre are 0.7766 / 0.6777. The promotion-line figures above are also legacy calibre (fourway flat: 0.8444 / 0.7106 — see gedi `BOP_OFFICIAL_BASELINES.md`). Run-plan titles (#1/#2) and archive paths below cite the same legacy figures. Grasp rows #5/#6 use the archived grasp script's per-object-median statistics, unchanged. From #23 onward `popoe.metrics` reports flat calibre by default (legacy value kept as a trailing diagnostic line).

| # | Experiment | Archive number (source) | popoe entrypoint | Class | Reproduced | popoe commit / pod / date | Status |
|---|---|---|---|---|---|---|---|
| 1 | YCB-V full BOP AR | **0.7668** — `score_rules_ycbvg32m`; recipe: CNOS-FastSAM TOPK2 + gripper label pooling + grid-32 + O3D + fit×s_feat_1(×metric) | `examples/bop_eval.py --bop $BOP/ycbv --detections data/detections/cnos/cnos-fastsam_ycbv-test.json --merge ycbv --topk 2 --grid 32 --solver o3d --weights 1.0,0.7,0.5,0.3,0.2 --render-backend nvdiffrast --out … --cache … --cand-csv …` (full cmd → Run plan #1) | **GPU-POD** | **0.7781** (MSSD 0.7934 / MSPD 0.7414 / VSD 0.7995) | `75553a1` / `wrmy8k0thtxjq6` / 2026-07-26 | ☑ Δ=+0.0113 |
| 2 | LM-O full BOP AR | **0.6726** — `lmog32`; CNOS∪SAM6D union detections + same pipeline | `examples/bop_eval.py --bop $BOP/lmo --sources cnos=…/cnos-fastsam_lmo-test.json,sam6d=…/sam6d_ism_lmo.json --merge none --topk 2 --grid 32 --solver o3d …` (full cmd → Run plan #2) | **GPU-POD** | **0.6792** (MSSD 0.7242 / MSPD 0.7566 / VSD 0.5568) | same campaign | ☑ Δ=+0.0066 |
| 3 | YCB-V AR(2/3) | 0.7528 (same run as #1) | same pose CSV as #1; score with gedi / freezev2 `freezev2_compute_ar_mssd_mspd.py` | **LOCAL-CPU** (post #1) | **0.7674** | same | ☑ Δ=+0.0146 |
| 4 | LM-O AR(2/3) | 0.7324 (same run as #2) | same pose CSV as #2; same AR scorer as #3 | **LOCAL-CPU** (post #2) | **0.7404** | same | ☑ Δ=+0.0080 |
| 5 | YCB-V grasp ADD(-S)@0.1d | **0.8173** (median 2.5 mm / 6.6°) — gedi `scripts/freezev2_grasp_eval.py` | external grasp CLI on #1 CSV | **LOCAL-CPU** (post #1) | **0.8240** (@0.05d 0.7716; med 2.5 mm / 11.9°) | same | ☑ Δ=+0.0067 |
| 6 | LM-O grasp ADD(-S)@0.1d | **0.7617** (7.2 mm / 5.8°) | same as #5 on #2 CSV | **LOCAL-CPU** (post #2) | **0.7706** (@0.05d 0.5146; med 7.1 mm / 5.9°) | same | ☑ Δ=+0.0089 |

## BOP-Classic-Core seven-set ledger (2026-07-27, OFFICIAL SERVER SCORES)

> The seven core datasets scored by the **BOP evaluation server**, method
> `popoe-cnos`, popoe **`0c93d3e`**, pod `bidtug84rly2xo` (4090, stopped).
> Submissions 39689–39695, all kept **private**. These are not local
> measurements — the server computed them from the uploaded pose CSVs.
> Recipe: official CNOS-FastSAM default detections, **single source** (same
> footing as the official FreeZe(CNOS) row); `--merge ycbv` on YCB-V, `auto`
> elsewhere. Artefacts: `../gedi/ycbv_local_data/bop7_full_20260727/`.

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
> Evidence, null control and per-object breakdown:
> `../gedi/BOP_OFFICIAL_BASELINES.md` 乙-5; artifacts
> `../gedi/ycbv_local_data/vcolor_ab_20260728/`.

### Campaign notes (2026-07-26)

- Fresh clone path on pod: `/workspace/popoe_verify_75553a1` @ `75553a1`.
- Outputs: `outputs/pipeline_verify_20260726/{parity_ycbv_g32m,parity_lmo_g32_union}.csv` (+ AR/VSD/grasp logs, `master.log`, `CAMPAIGN_DONE`).
- Scheduling: O3D is CPU-bound; `OMP_NUM_THREADS=16` capped thrash; brief dual-GPU occupancy then serial finish.
- Multi-mask LM-O union is exercised by #2 (secondary row can be read as covered for full-AR end-to-end).

## Contribution-level parity (secondary)

| Experiment | Archive result | popoe entrypoint | Class | Status |
|---|---|---|---|---|
| Adaptive visual weight | beats best-fixed on all 4 datasets | Built into `bop_eval.py --weights 1.0,0.7,0.5,0.3,0.2` (ChampionScorer per-target argmax over w). Cross-dataset 4-set claim still needs TUD-L / IC-BIN BOP data + GPU runs (not in this repo). Offline post-hoc: gedi `scripts/freezev2_adaptive_select.py` over per-w CSVs. | **GPU-POD** (YCB-V/LM-O covered by #1/#2); **GAP** for TUD-L/IC-BIN data | ☐ |
| Canonical-space scoring | 26-rule ablation; champion rule constant across datasets | Live rule = `ChampionScorer` (`s_icp * max(s_feat_1,0) * metric_fit?`). Offline re-sweep: `examples/rule_replay.py <cand.csv> --target-csv <poses.csv> --rule "s_icp*s_feat_1" --rule "s_icp*s_feat_1*metric_fit" --out-dir …` on a `--cand-csv` dump from #1/#2. `--target-csv` defines the full target universe and zero-pads detector misses; without it, output AR is a candidate-bearing ceiling. Existing historical cands under `../gedi/ycbv_local_data/union_scoring_20260716/` may be used only when paired with a matching full target CSV. | **LOCAL-CPU** (once cand-csv exists) | ☐ |
| Gripper label pooling + metric_fit | obj20 +33.6 pt; 2×2 ablation | `bop_eval.py --merge ycbv` (pools 19:20, size_aware metric_fit) vs `--merge none` on YCB-V objs 19,20 (`--objs 19,20`). Live scorer: `ChampionScorer(size_aware=True)` for pooled pairs. | **GPU-POD** (subset ablation) | ☐ |
| Multi-mask / detection union (LM-O) | CNOS∪SAM6D +2.8 pt | Smoke (no GPU): `examples/union_smoke.py --dataset lmo --source sam6d=data/detections/sam6d/sam6d_ism_lmo.json`. Full AR: same as headline #2 (`--sources cnos=…,sam6d=…`). CNOS-only control: `--detections …/cnos-fastsam_lmo-test.json`. | **LOCAL-CPU** smoke + **GPU-POD** full | ☑ full-AR via #2 (2026-07-26); no separate CNOS-only control re-run |

## Rules of engagement

1. **Fresh clone only.** Every pod run executes popoe from a `git clone` at a
   recorded commit — never hand-`scp`'d single files. (The gedi campaign lost
   time to stale bare-name module copies living only on the pod; that failure
   class ends here.)
2. Raw per-image CSVs from parity runs stay out of git (`data/` is ignored);
   this ledger records the number + commit + pod. Copy CSVs worth keeping to
   the gedi archive under `ycbv_local_data/` with a dated subdir.
3. One row per run: if a re-run disagrees with the archive beyond tolerance,
   do not overwrite — add a row and investigate before promoting either
   number.

## Gap list (archive capabilities popoe does not have yet)

Full gap report: `../gedi/EXPERIMENTS.md` Appendix B. Known candidates:

- ~~Grasp-axis evaluation (ADD(-S)@0.1d)~~ — **CLOSED 2026-08-06**:
  ported to `src/popoe/metrics/grasp.py` (`python -m popoe.metrics.grasp
  preds.csv`, env `BOP_PATH` + `POPOE_BOP_TOOLKIT`). Parity verified
  **bit-identical** against the archived originals
  (`outputs/pipeline_verify_20260726/parity_{ycbv,lmo}_grasp.txt`: YCB-V
  0.8240/0.7716, LM-O 0.7706/0.5146). Keeps the archived per-object calibre
  by design — see the module docstring before "fixing" it to flat.
- ~~VSD computation / AR(2/3)~~ — **CLOSED (stale entry, corrected
  2026-08-06)**: `src/popoe/metrics/vsd.py` (nvdiffrast, no OpenGL/EGL) and
  `src/popoe/metrics/ar.py` are the ports of gedi
  `freezev2_vsd_compute.py` / `freezev2_compute_ar_mssd_mspd.py`, both
  reporting BOP flat calibre with legacy per-object as secondary
  (`metrics/aggregate.py`). This entry predated the metrics package.
- Adaptive visual-weight sweep harness (`freezev2_sweep_vis_weight.py`) —
  partially absorbed by `bop_eval --weights` + ChampionScorer; multi-dataset
  adaptive claim still needs TUD-L / IC-BIN.
- Full BOP RGB-D + CAD models for YCB-V / LM-O — **not** complete in the
  local archive (`../gedi/ycbv_local_data/bop_data/` is GT-meta + partial
  RGB for offline metrics). Full image trees live on the pod volume
  (`/workspace/bop_data/{ycbv,lmo}`). **Confirmed by Vincent 2026-07-22:
  the network volume (8rf4r42sf1) retains the full data tree and envs —
  mounting it is sufficient, no re-download needed. This gap is closed for
  pod runs.**

## Run plan (offline prep, 2026-07-22)

Path aliases used below (resolve on the machine you run on):

| Alias | Local (this workstation) | Pod (typical) |
|---|---|---|
| `$POPOE` | `/home/vincent/work/popoe` (fresh clone on pod) | `/workspace/popoe` |
| `$GEDI` | `/home/vincent/work/gedi` | `/workspace/gedi` or N/A |
| `$BOP` | **GAP** for full RGB-D — local has only `../gedi/ycbv_local_data/bop_data/` (GT + models_eval + sparse RGB) | `/workspace/bop_data` |
| `$DET` | `$POPOE/data/detections` (present locally; verified 2026-07-22) | copy from clone or volume |
| `$OUT` | N/A for GPU | `/workspace/results/parity_20260722` (create fresh) |

Env for GPU feature extraction (pod): `POPOE_GEDI_PATH`, `POPOE_BOP_TOOLKIT`,
CUDA + nvdiffrast (official numbers used `--render-backend nvdiffrast`). Fresh
`git clone` of popoe at a recorded commit — never scp single files.

Detection files verified present under `$POPOE/data/detections/`:

- `cnos/cnos-fastsam_ycbv-test.json`, `cnos/cnos-fastsam_lmo-test.json`
- `nids/nids_wa_sappe_ycbv.json`, `nids/nids_wa_sappe_lmo.json` (promotion line; not needed for gedi-headline #1/#2)
- `sam6d/sam6d_ism_ycbv.json`, `sam6d/sam6d_ism_lmo.json`, `sam6d/union_cnos_sam6d_lmo.reference.json`

### #1 — YCB-V full BOP AR 0.7668 (GPU-POD)

- **Class**: GPU-POD
- **Command** (from `$POPOE`, pod):

```bash
mkdir -p "$OUT"
uv run python examples/bop_eval.py \
  --bop "$BOP/ycbv" \
  --detections "$DET/cnos/cnos-fastsam_ycbv-test.json" \
  --merge ycbv \
  --topk 2 \
  --grid 32 \
  --solver o3d \
  --weights 1.0,0.7,0.5,0.3,0.2 \
  --render-backend nvdiffrast \
  --out "$OUT/parity_ycbv_g32m.csv" \
  --cache "$OUT/cache_ycbv_g32m" \
  --cand-csv "$OUT/parity_ycbv_g32m_cands.csv"
```

- **Data deps**: full YCB-V BOP test RGB-D + `models/` + `test_targets_bop19.json`
  (**pod only**); CNOS-FastSAM JSON (local OK under `$DET/cnos/`).
- **Post (LOCAL-CPU or pod CPU)**: AR(2/3) + VSD → full BOP AR:

```bash
BOP_PATH="$BOP/ycbv" python "$GEDI/scripts/freezev2_compute_ar_mssd_mspd.py" \
  "$OUT/parity_ycbv_g32m.csv"
# VSD (needs models + depth; typically pod):
python "$GEDI/scripts/freezev2_vsd_compute.py" "$OUT/parity_ycbv_g32m.csv"
```

- **Preconditions**: nvdiffrast on matching GPU arch (4090 sm_89); GeDi + DINOv2
  loadable; resume-safe if `--out` partially written.
- **Est. GPU wall**: ~15–22 h on RTX 4090 for full 21-obj set (formal union2
  fullset wall was ~22 h; CNOS-only is lighter on candidates, still O(10+ h)).
  Optional 2-way split: `--objs 1,2,…,10` / `11,…,21` then merge CSVs.

### #2 — LM-O full BOP AR 0.6726 (GPU-POD)

- **Class**: GPU-POD
- **Command**:

```bash
uv run python examples/bop_eval.py \
  --bop "$BOP/lmo" \
  --sources "cnos=$DET/cnos/cnos-fastsam_lmo-test.json,sam6d=$DET/sam6d/sam6d_ism_lmo.json" \
  --merge none \
  --topk 2 \
  --grid 32 \
  --solver o3d \
  --weights 1.0,0.7,0.5,0.3,0.2 \
  --render-backend nvdiffrast \
  --out "$OUT/parity_lmo_g32_union.csv" \
  --cache "$OUT/cache_lmo_g32" \
  --cand-csv "$OUT/parity_lmo_g32_union_cands.csv"
```

- **Data deps**: full LM-O BOP test (**pod**); CNOS + SAM6D JSON (local OK).
- **Post**: same AR/VSD scripts with `BOP_PATH=$BOP/lmo`.
- **Preconditions**: same as #1. Do **not** pass `--use-s-coarse` (hurts LM-O;
  that is the popoe promotion line, not the gedi headline).
- **Est. GPU wall**: ~3–5 h on 4090 (union3 L40S log ~2.7 h for related run).

### #3 / #4 — AR(2/3) (LOCAL-CPU after #1 / #2)

- **Class**: LOCAL-CPU (once pose CSVs exist)
- **Command**: see Post blocks under #1 / #2 (`freezev2_compute_ar_mssd_mspd.py`).
- **Data deps**: pose CSV + `$BOP/{ycbv,lmo}` GT (`scene_gt.json`,
  `models_eval/`). Local archive GT path usable:
  `../gedi/ycbv_local_data/bop_data/{ycbv,lmo}/` (verified present).
- **Est.**: minutes on CPU; no GPU.

### #5 / #6 — Grasp ADD(-S)@0.1d (LOCAL-CPU after #1 / #2; GAP port)

- **Class**: LOCAL-CPU post-pose; **GAP** = no popoe-native grasp CLI
- **Command** (gedi script; path hardcoded default `/workspace/bop_toolkit`
  — set `PYTHONPATH` / edit sys.path or run where toolkit lives):

```bash
# YCB-V (#5) — target archive 0.8173
BOP_PATH="$BOP/ycbv" python "$GEDI/scripts/freezev2_grasp_eval.py" \
  "$OUT/parity_ycbv_g32m.csv"

# LM-O (#6) — target archive 0.7617
BOP_PATH="$BOP/lmo" python "$GEDI/scripts/freezev2_grasp_eval.py" \
  "$OUT/parity_lmo_g32_union.csv"
```

- **Data deps**: pose CSV + `models_eval/*.ply` + per-scene `scene_gt.json`
  (local `../gedi/ycbv_local_data/bop_data/` sufficient for metrics if scenes
  in the CSV are covered).
- **Est.**: <5 min CPU each; zero GPU.
- **Sanity without new poses**: recompute on existing gedi champion CSV under
  `../gedi/ycbv_local_data/freezev2/score_rules_ycbvg32m/rule_champion_size.csv`
  (archive path for 0.7668 / grasp 0.8173 chain).

### C1 — Adaptive visual weight (GPU-POD partial / GAP multi-dataset)

- **Class**: GPU-POD for YCB-V/LM-O (already inside #1/#2 via `--weights`);
  **GAP** for TUD-L + IC-BIN (no data in this workspace).
- **Command**: no extra live flag — ChampionScorer selects best w per target.
  Offline histogram over fixed-w CSVs (gedi):
  `python "$GEDI/scripts/freezev2_adaptive_select.py" out.csv w1.csv w0.7.csv …`
- **Est. GPU**: covered by #1/#2 wall time.

### C2 — Canonical-space scoring / 26-rule replay (LOCAL-CPU)

- **Class**: LOCAL-CPU
- **Command** (popoe column names: `s_icp`, `s_feat_1`, `metric_fit`, optional
  `s_coarse` — **not** the older gedi `icp_fit` header):

```bash
uv run python examples/rule_replay.py \
  ../gedi/ycbv_local_data/union_scoring_20260716/popoe_ycbv_formal_A_cands.csv \
  --rule "s_icp*s_feat_1" \
  --rule "s_icp*s_feat_1*metric_fit" \
  --rule "s_icp*s_feat_1*metric_fit*s_coarse" \
  --out-dir /tmp/popoe_rule_replay_ycbv
```

- **Data deps**: any popoe `--cand-csv` dump (existing union_scoring cands
  verified under `../gedi/ycbv_local_data/union_scoring_20260716/`).
- **Est.**: <1 min CPU. Full 26-rule grid is the same tool with more `--rule`s.

### C3 — Gripper pooling 2×2 (GPU-POD subset)

- **Class**: GPU-POD
- **Command** (minimal ablation on clamps only):

```bash
# pool + metric_fit (default merge=ycbv → size_aware on 19/20)
uv run python examples/bop_eval.py \
  --bop "$BOP/ycbv" \
  --detections "$DET/cnos/cnos-fastsam_ycbv-test.json" \
  --objs 19,20 --merge ycbv --topk 2 --grid 32 --solver o3d \
  --render-backend nvdiffrast \
  --out "$OUT/ab_clamp_merge.csv" --cache "$OUT/cache_clamp"

# no pool control
uv run python examples/bop_eval.py \
  --bop "$BOP/ycbv" \
  --detections "$DET/cnos/cnos-fastsam_ycbv-test.json" \
  --objs 19,20 --merge none --topk 2 --grid 32 --solver o3d \
  --render-backend nvdiffrast \
  --out "$OUT/ab_clamp_nopool.csv" --cache "$OUT/cache_clamp"

# lab path (NOT headline): mask-stage nearest size_select on top of merge
# (opt-in; default --size-select none). Needs depth on the BOP Scene.
uv run python examples/bop_eval.py \
  --bop "$BOP/ycbv" \
  --detections "$DET/cnos/cnos-fastsam_ycbv-test.json" \
  --objs 19,20 --merge ycbv --size-select nearest --topk 2 --grid 32 \
  --render-backend nvdiffrast \
  --out "$OUT/ab_clamp_size_select.csv" --cache "$OUT/cache_clamp"
# equivalent library helper: popoe.freeze.recipes.ycbv_lab_segmentor(...)
```

- **Est. GPU**: ~1–2 h (300 targets × 2 configs) on 4090; +1 h for size-select lab run.
- **Offline (LOCAL-CPU)** dual-CAD / metric_fit selection A/B on an existing
  cand dump (no re-encode): `scripts/eval_dual_cad_metric_fit_ab.py` — see
  `outputs/dual_cad_metric_fit_ab/` (2026-07-26: dual assignment lifts scene-48
  obj20 ADD-S@0.1d 0.373→0.747 vs independent `metric_fit`; **AR(2/3) on
  obj19+20: dual 0.800 / no_mf 0.777 / with_mf 0.725**).
- **Live dual-CAD** (GPU, score-affecting, default off):

```bash
uv run python examples/bop_eval.py \
  --bop "$BOP/ycbv" --detections "$DET/cnos/cnos-fastsam_ycbv-test.json" \
  --objs 19,20 --merge ycbv --dual-assign --topk 2 --grid 32 \
  --render-backend nvdiffrast \
  --out "$OUT/ab_clamp_dual.csv" --cand-csv "$OUT/ab_clamp_dual_cands.csv" \
  --cache "$OUT/cache_clamp"
```

  Library: `popoe.confusable_select`. Mask-stage companion: `--size-select nearest`.

### C4 — Multi-mask union LM-O (LOCAL-CPU smoke + GPU-POD full)

- **Class**: LOCAL-CPU smoke; GPU-POD full (= #2)
- **Smoke**:

```bash
uv run python examples/union_smoke.py --dataset lmo \
  --source sam6d=data/detections/sam6d/sam6d_ism_lmo.json
uv run python examples/union_smoke.py --dataset ycbv \
  --source sam6d=data/detections/sam6d/sam6d_ism_ycbv.json
```

- **Full AR**: command under #2; CNOS-only control uses `--detections` single file.
- **Est. GPU**: same as #2.

### Pod session budget (all GPU-POD items in one session)

| Item | Est. GPU h (4090) |
|---|---|
| #1 YCB-V full parity | 15–22 |
| #2 LM-O full parity | 3–5 |
| C3 clamp 2×2 (optional same session) | 1–2 |
| VSD post (#1+#2) | ~0.3–0.5 |
| **Total** | **~20–30 h** |
| **Cost @ $0.69/hr** | **~$14–21** |

LOCAL-CPU items (#3–#6, C2, C4 smoke, pytest) add negligible $ and can run
on this workstation after CSVs land (or on the pod after GPU finishes).

## Offline verification log

Prep date: **2026-07-22**. Host: local workstation (no NVIDIA driver —
`nvidia-smi` failed; no GPU smoke of feature stack). Scope: path existence,
CLI flags from code, light CPU tests. GPU parity numbers still ☐.

### Path / artifact checks (read-only)

| Path | Role | Present? |
|---|---|---|
| `data/detections/cnos/cnos-fastsam_{ycbv,lmo}-test.json` | #1/#2 detections | yes (ls 2026-07-22) |
| `data/detections/sam6d/sam6d_ism_{ycbv,lmo}.json` | #2 union | yes |
| `data/detections/nids/nids_wa_sappe_{ycbv,lmo}.json` | promotion line only | yes |
| `../gedi/ycbv_local_data/freezev2/` | gedi g32 candidates + score_rules | yes |
| `../gedi/ycbv_local_data/union_scoring_20260716/` | popoe formal CSVs + cands + grasp logs | yes |
| `../gedi/ycbv_local_data/bop_data/{ycbv,lmo}/` | GT meta + models_eval (+ sparse RGB) | yes — **not** full BOP RGB-D |
| `../gedi/scripts/freezev2_grasp_eval.py` | #5/#6 external CLI | yes |
| `../gedi/scripts/freezev2_compute_ar_mssd_mspd.py` | #3/#4 AR(2/3) | yes |
| `../gedi/scripts/freezev2_vsd_compute.py` | full BOP AR VSD leg | yes |
| `/workspace/bop_data/{ycbv,lmo}` | full RGB-D for GPU | **not on this host** (pod volume) |
| CUDA / nvdiffrast | feature extraction | **unavailable locally** |

### CLI flags verified from code (not memory)

- `examples/bop_eval.py`: mutually exclusive `--detections` / `--sources`;
  defaults `--topk 2`, `--grid 32`, `--solver o3d`, `--merge ycbv`
  (2026-07-27: default is now `auto` — ycbv pooling on ycbv, none elsewhere;
  dataset layout from `--dataset`/`BOP_LAYOUTS`, missing images fatal unless
  `--allow-missing-images`, and server submissions need
  `examples/bop_time_normalize.py` on the raw CSV first),
  `--weights` = recipes.WEIGHTS `(1.0,0.7,0.5,0.3,0.2)`,
  `--render-backend nvdiffrast|trimesh|auto`; Champion rule via
  `ChampionScorer` / `stages_for_object` (see `src/popoe/scoring.py`,
  `src/popoe/freeze/recipes.py`).
- `examples/rule_replay.py`: product rules over cand-csv columns; zero GPU.
- `examples/union_smoke.py`: defaults CNOS+NIDS under `data/detections/`;
  `--source name=path` overrides/adds; no RGB-D required.
- `examples/pipeline_selfcheck.py` / `solver_swap_demo.py`: need CUDA + full
  BOP mesh/RGB — **not run** locally (GPU-POD / >10 min risk).

### Smoke commands (results filled after run)

| Command | Est. | Result (review run, 2026-07-22) |
|---|---|---|
| `uv run pytest tests/` | <5 min | **281 passed, 0 skipped**, 19 s (2026-07-27, local `.venv`). Was 120 on 2026-07-22. Zero skips needs the full local env: `torch` (CPU build), `open3d`, `pycocotools`, `opencv-python-headless`, `scipy`, `pandas`, `pytest`, plus source-built `teaserpp_python` — without torch and teaserpp the same suite reports 257 passed / 7 skipped. |
| `uv run python examples/union_smoke.py --dataset ycbv --source sam6d=data/detections/sam6d/sam6d_ism_ycbv.json` | <2 min | **OK** end-to-end (3-way union, 746 champions) |
| `uv run python examples/union_smoke.py --dataset lmo --source sam6d=data/detections/sam6d/sam6d_ism_lmo.json` | <2 min | **OK** end-to-end (393 champions) |
| `uv run python examples/rule_replay.py …/popoe_ycbv_formal_A_cands.csv --rule "s_icp*s_feat_1" --rule "s_icp*s_feat_1*metric_fit" --rule "s_icp*s_feat_1*metric_fit*s_coarse" --out-dir …` | <1 min | **OK** — 21 800 hyps / 1 669 targets; ×metric_fit flips 44.0% vs formal baseline; +s_coarse flips 0.2% (formal baseline is itself s_coarse-arbitrated — consistency check ✓). Original plan referenced `popoe_ycbv_union2_cands.csv` which does not exist; corrected to `popoe_ycbv_formal_A_cands.csv`. |
| `uv run python examples/pipeline_selfcheck.py …` | needs GPU | **skipped** (no local GPU) |
| full `bop_eval` parity | 15–22 h GPU | **done 2026-07-26** — see Headline ledger #1–#6 |

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
| `cnos-lab` / `cnos-live` | Self-built CNOS tracks (lab / live; `cnos-lab` formerly `cnos-v3`) — not paper headline |

### Remaining follow-up

| Item | Depends on | Notes |
|---|---|---|
| G3 `muse-repro` pose-impact rerun | PR #10 | Optional; segmentation AP is ledgered above, but pose promotion still uses official four-way `muse` JSON |
| Re-run official-source ledger under merged evaluator | PR #5 | Optional confirmation |
| Human review panels | `scripts/export_bop_seg_review.py` | CPU; works on existing JSONs |

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

**Not a parity row and not a performance claim.** ARCHITECTURE.md's
Pluggability section quotes these to rank three `PoseSolver` implementations
against *each other* through one identical chain; it states outright that the
table "is not a performance claim for popoe". `solver_swap_demo` is not the
evaluated pipeline (it runs `FreeZeScorer` on GT masks at fixed thresholds) and
obj 5 is a known-weak registration case — `recall@0.1d` is 0.000 for all three.
Ledgered here because the previous version of this table was withdrawn for
being unverifiable (ISSUES.md 2026-07-26), so its replacement should be
reachable from the ledger rather than from prose alone.

Metric: MSSD via bop_toolkit `pose_error.mssd`, symmetries expanded from
`models_eval/models_info.json` (obj 5 -> 1 transform, identity: BOP declares the
mustard bottle NOT symmetric). Seeded, `--seed 42`. 140 of obj 5's 150 test
instances — the pod died at 140; the missing 10 are all scene 52, so the
population is scene 50 complete + scene 52 partial.

| Solver | median MSSD | @0.2d | @0.5d | popoe commit | Artefacts | Verdict |
|---|---:|---:|---:|---|---|---|
| `RansacSolver` (freeze_ransac) | **42.9 mm** (0.218 d) | **0.371** | **0.600** | `63c5e7d` | `outputs/solver_swap_20260726/` | best of the three |
| `Open3DFeatureRansacSolver` 1-shot | 111.2 mm (0.566 d) | 0.271 | 0.457 | same | same | baseline for the composition |
| `Open3DFeatureRansacSolver(n_restarts=8)` | 62.7 mm (0.319 d) | 0.343 | 0.521 | same | same | composition helps, parity NOT reached |

Reading: handing several hypotheses to the scorer nearly halves 1-shot's median
MSSD and wins head-to-head 72:33 (35 tied), closing roughly two thirds of the
gap to `freeze_ransac` — but not reaching it. Ordering is stable at every
threshold.

Artefacts: `outputs/solver_swap_20260726/` (gitignored, local + pod) holds
`mssd140_run.log` with all 140 per-instance rows, `mssd140_PROVENANCE.txt`
(commit, pod `oqxijvj1cytc7m`, fresh-clone path, exact cmd), the earlier
`seeded150_*` and `unseeded5_*` runs, and a README recording the withdrawal.
The run log has **no summary block** — the pod died before it printed — so
every figure above was derived post-hoc from the per-instance rows.

**Recompute check (2026-07-27, LOCAL-CPU):** all three medians, all six
recalls, `recall@0.1d = 0.000` for all three, and the 72/33/35 head-to-head
reproduce exactly from `mssd140_run.log`. `solver_swap_demo` now prints the
`@0.5d` column itself (`698ffd5`); before that it emitted only
@0.05/0.1/0.2d and this row's `@0.5d` had to be re-derived.

## Phase E seven-set freeze tables (frozen 2026-08-06, run NOTHING before both are pinned)

Two independent breadth lines (gedi `EXPERIMENT_PLAN.md` §5, Vincent
2026-08-06). Both frozen BEFORE any new server score is read; no post-hoc
selection between lines. Method identity for both = the Phase D frozen recipe
flags at tag `twoline-rerank-fix-20260731` (`509072e`); if a new tag is cut,
RE-FREEZE these tables against it before running.

Common pins (both lines): `--topk 2 --grid 32 --solver o3d --seed 42
--weights 1.0,0.7,0.5,0.3,0.2 --render-rerank --render-backend nvdiffrast`;
YCB-V uses `--merge ycbv --use-s-coarse`; all other datasets `--merge none`,
no `--use-s-coarse`. OMP/host/sharding are runtime provenance only.

### E-cnos (`tuned-cnos` x 7)

Comparator: FreeZe(CNOS) per-set (same detection input). LM-O + YCB-V were to
reuse the accepted Phase D B-single results (identical inputs) — **reuse
condition void: all eight Phase D server scores were voided pending re-run;
reuse LM-O/YCB-V only after the Phase D re-run is accepted.**
Five new runs for the other sets.

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
submissions for ALL SEVEN sets, method 441 "SAM6D"** (Vincent 2026-08-06;
amended same day from 546 FastSAM(RGB) BEFORE any server score — 441 is the
strongest official SAM6D variant, mean seg AP 0.481 vs 546's 0.428, board
family spread 5.3pt; seg batch 6965-6971, 2023-12-05; 441's method page mixes
THREE tasks — seg, 2D detection, and 6D localization batches — the seg batch
was identified by matching per-set AP against the leaderboard row). A
DIFFERENT detection identity from Phase D `tuned-4way` (local ISM); all seven
sets run fresh, and Phase D vs Phase E numbers must never be presented as
same-input.

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

Preflight per dataset (EXPERIMENT_PLAN §5.3): layout/detection checksum match
against this table -> numeric smoke -> full run -> acceptance -> time-normalized
private submission. Stop rules in EXPERIMENT_PLAN §7 apply unchanged.
