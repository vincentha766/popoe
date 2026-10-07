# scripts/

Diagnostics and ablation drivers. None of these is the evaluated BOP entry point; that is `examples/bop_eval.py`. Most require a completed CSV, a `--cand-csv` dump, or a BOP split.

## Identity and gates

| Script | What it does |
|--------|----------------|
| `freeze_detections.py` | Materialize or verify `data/detections/` against `MANIFEST.sha256` (no symlinks; hashes bound to paths) |
| `make_gt_detections.py` | Emit BOP `mask_visib` as a detections JSON: a perception upper bound, not a submittable result |
| `check_rerank_symmetry.py` | Fail if `--render-rerank` inflates flip `s_icp` |

## Segmentation A/B (CPU)

| Script | What it does |
|--------|----------------|
| `eval_cnos_size_gate_ab.py` | CNOS-lab depth size gate vs appearance top-1 (YCB-V clamps) |
| `eval_cnos_size_select_ab.py` | Nearest-diameter and soft size-select follow-up |
| `eval_dual_cad_metric_fit_ab.py` | Offline clamp assignment from a `--cand-csv` (`metric_fit`) |
| `export_bop_seg_review.py` | RGB and mask review panels from a detections JSON |
| `export_faithful4way_seg_review.py` | The same candidate set `bop_eval` sees under the 4-source union |

## Pose A/B (usually GPU, plus a prior run)

| Script | What it does |
|--------|----------------|
| `coarse_vs_refined.py` | Champion pre-ICP vs post-ICP poses from a `--cand-csv` |
| `pose_ab_instances.py` | Per-instance AR delta: which instances a pose change improved and which it regressed |
| `flip_rescore_ab.py` | Whether the live score prefers an explicit 180° flip |
| `perview_probe.py` | Per-view DINOv2 matching vs the stored view-mean |
| `sar_render_compare.py` | Symmetry-aware render re-rank vs the live score |

## Ablation shells

| Script | What it does |
|--------|----------------|
| `ablation_geom_backbone.sh` | Visual-only / GeDi / FPFH geometric-branch ablation |
| `run_fpfh_ablation_pod.sh` | Driver for the same ablation on a GPU host |
| `fidelity_ab_run.sh` | Cost of `--icp-dense` and `--tau-diameter` measured separately |
| `vcolor_ab_run.sh` | Vertex-colour CAD shading vs UV-only |

Runnable entry points are in `examples/` and indexed in [README.md](../README.md#layout).
