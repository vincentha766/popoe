# scripts/

Diagnostics and ablation drivers. They are **not** the evaluated BOP entry — that is `examples/bop_eval.py`. Most need a finished CSV, `--cand-csv` dump, or a BOP split.

## Identity / gates

| Script | What it does |
|--------|----------------|
| `freeze_detections.py` | Materialize/verify `data/detections/` against `MANIFEST.sha256` (no symlinks, hashes bound to paths) |
| `make_gt_detections.py` | Emit BOP `mask_visib` as a detections JSON (perception upper bound; not submittable) |
| `check_rerank_symmetry.py` | Fail if `--render-rerank` inflates flip `s_icp` (the 2026-07-30 defect) |

## Segmentation A/B (CPU)

| Script | What it does |
|--------|----------------|
| `eval_cnos_size_gate_ab.py` | CNOS-lab depth size gate vs appearance top-1 (YCB-V clamps) |
| `eval_cnos_size_select_ab.py` | Nearest-diameter / soft size-select follow-up |
| `eval_dual_cad_metric_fit_ab.py` | Offline clamp assignment from a `--cand-csv` (`metric_fit`) |
| `export_bop_seg_review.py` | RGB/mask review panels from detections JSON |
| `export_faithful4way_seg_review.py` | Same candidate set `bop_eval` uses under the 4-source union |

## Pose A/B (usually GPU + a prior run)

| Script | What it does |
|--------|----------------|
| `coarse_vs_refined.py` | Champion pre-ICP vs post-ICP poses from `--cand-csv` |
| `pose_ab_instances.py` | Per-instance AR delta: what a pose change fixed vs broke |
| `flip_rescore_ab.py` | Whether the live score prefers an explicit 180° flip |
| `perview_probe.py` | Per-view DINOv2 matching vs the stored view-mean |
| `sar_render_compare.py` | Symmetry-aware render re-rank vs the live score |

## Campaign shells

| Script | What it does |
|--------|----------------|
| `ablation_geom_backbone.sh` | Visual-only / GeDi / FPFH geometric-branch ablation |
| `run_fpfh_ablation_pod.sh` | Driver for that ablation on a GPU box |
| `fidelity_ab_run.sh` | `--icp-dense` and `--tau-diameter` priced separately |
| `vcolor_ab_run.sh` | Vertex-colour CAD shading vs the old UV-only beige |

Related examples (not in this directory): `examples/union_smoke.py`, `examples/bop_seg_eval.py`, `examples/rule_replay.py`, `examples/pipeline_selfcheck.py`.
