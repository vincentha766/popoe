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

## Headline ledger

| # | Experiment | Archive number (source) | popoe entrypoint | Reproduced | popoe commit / pod / date | Status |
|---|---|---|---|---|---|---|
| 1 | YCB-V full BOP AR | **0.7668** — `score_rules_ycbvg32m`; recipe: CNOS-FastSAM TOPK2 + gripper label pooling + grid-32 + O3D + fit×s_feat_1(×metric) | `examples/bop_eval.py` (flags TBD) | — | — | ☐ |
| 2 | LM-O full BOP AR | **0.6726** — `lmog32`; CNOS∪SAM6D union detections + same pipeline | `examples/bop_eval.py` | — | — | ☐ |
| 3 | YCB-V AR(2/3) | 0.7528 (same run as #1) | same run as #1 | — | — | ☐ |
| 4 | LM-O AR(2/3) | 0.7324 (same run as #2) | same run as #2 | — | — | ☐ |
| 5 | YCB-V grasp ADD(-S)@0.1d | **0.8173** (median 2.5 mm / 6.6°) — gedi `scripts/freezev2_grasp_eval.py`, locally recomputable from pose CSVs | TBD: port grasp-axis eval or run gedi script on popoe pose CSVs | — | — | ☐ |
| 6 | LM-O grasp ADD(-S)@0.1d | **0.7617** (7.2 mm / 5.8°) | same as #5 | — | — | ☐ |

## Contribution-level parity (secondary)

| Experiment | Archive result | popoe entrypoint | Status |
|---|---|---|---|
| Adaptive visual weight | beats best-fixed on all 4 datasets | TBD | ☐ |
| Canonical-space scoring | 26-rule ablation; champion rule constant across datasets | TBD | ☐ |
| Gripper label pooling + metric_fit | obj20 +33.6 pt; 2×2 ablation | TBD | ☐ |
| Multi-mask / detection union (LM-O) | CNOS∪SAM6D +2.8 pt | `examples/union_smoke.py` → full run TBD | ☐ |

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

- Grasp-axis evaluation (ADD(-S)@0.1d) — currently only in gedi
  `scripts/freezev2_grasp_eval.py`.
- VSD computation / cross-check tooling (`freezev2_vsd_*.py`).
- Adaptive visual-weight sweep harness (`freezev2_sweep_vis_weight.py`).
