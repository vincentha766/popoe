# popoe — Pipeline Of Pose Estimation

A modular **6-DoF object pose** framework, evaluated on **BOP**. Stages sit behind small `Protocol` contracts so a segmentor, backbone, solver, or scorer can grow without rewriting the rest.

**Scope**: pose library — BOP datasets, metrics, evaluated recipes. Grasping, HTTP, and robot stacks belong elsewhere and should call these contracts.

```
ObjectModel (CAD) ─┬─ QueryEncoder ──────────── q, CanonFrame ─┐
                   ├─ Segmentor ─ Detection ─┐                 │
Scene (RGB-D, K) ──┴─────────────────────────┴─ TargetEncoder ─┴─ PoseSolver ─ PoseRefiner* ─ PoseScorer ─ Selector ─ (R, t)
```

The reference method is FreeZe-v2-style (DINOv2 + GeDi → RANSAC → ICP → symmetry-aware scoring) with several `PoseSolver` implementations.

> Research code, `v0.1`. Contracts + fusion are CPU-tested. The reference run needs CUDA, GeDi, **nvdiffrast**, a BOP split, and detection JSONs (none of those ship in a clone). Design: [ARCHITECTURE.md](ARCHITECTURE.md).

### After a clone

| You want | Install | Also needed | Run |
|----------|---------|-------------|-----|
| Read / write a stage | `pip install -e .` | — | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Check the clone | `pip install -e ".[dev]"` (CI: `".[dev,reference]"`) | — | `pytest tests/` (CPU; GPU / nvdiffrast / GeDi skip) |
| Run 6-DoF on BOP | `pip install -e ".[reference]"` | CUDA, GeDi, **nvdiffrast**, BOP split, detection JSONs | [`examples/bop_eval.py`](#minimal-bop-eval) |

`PoseMethod.run(scene, obj)` is the library entry. `popoe.freeze.recipes.make_correspondence_pipeline` returns a `Pipeline`; `DirectPoseMethod` is the estimator graph (e.g. SAM-6D PEM). `examples/bop_eval.py` writes BOP CSVs via `correspond_pair`. `examples/solver_swap_demo.py` ranks solvers on **GT instances** — not detections, not the BOP loop.

## Install

```bash
pip install -e .                # numpy + scikit-learn
pip install -e ".[reference]"   # torch, open3d, trimesh, opencv, …
pip install -e ".[dev]"         # + pytest
pytest tests/                   # CPU; skips GPU / OpenCV / Open3D when missing
```

### External dependencies (not on PyPI)

Clone these yourself and **export the env vars**. Unset means empty — no implicit host path.

| Component | Env var | Notes |
|-----------|---------|-------|
| GeDi | `POPOE_GEDI_PATH` | [fabiopoiesi/gedi](https://github.com/fabiopoiesi/gedi) with `data/chkpts/3dmatch/chkpt.tar` |
| SAM 2 | `POPOE_SAM2_CKPT` | Dir with `sam2.1_hiera_large.pt`. Live SAM2 / MUSE / CNOS-lab only |
| bop_toolkit | `POPOE_BOP_TOOLKIT` | [thodan/bop_toolkit](https://github.com/thodan/bop_toolkit) — `solver_swap_demo.py`, `python -m popoe.metrics.ar` |
| nvdiffrast | — | **Required** for default `bop_eval.py` and `best_encoders()` (`nvdiffrast`). Missing it is an error, not a trimesh swap. Pass `--render-backend trimesh` or `auto` only if you accept different CAD views |
| CNOS / NIDS / SAM-6D producers | `POPOE_CNOS_PATH`, `external/NIDS-Net`, `POPOE_SAM6D_PATH` | Optional. Evaluated runs consume **already-written** files |

DINOv2 comes from `torch.hub` (`TORCH_HOME`). Licenses: [NOTICE](NOTICE) — **each upstream keeps its own; verify before use.**

```bash
# optional; skip for file-backed eval
git submodule update --init --recursive external/cnos external/NIDS-Net external/SAM-6D
```

Encode knobs (`POPOE_QUERY_POINTS`, `POPOE_TARGET_GRID`, `POPOE_DINO_LAYER`, `POPOE_TWO_SCALE_GEDI`, `POPOE_VIS_DIM`, `POPOE_GEOM_BACKBONE`, …) are recorded in the eval cache key. Changing one without a new `--cache` reuses stale features.

### BOP data

Download a Classic-Core split from [bop.felk.cvut.cz/datasets](https://bop.felk.cvut.cz/datasets/). `--bop` is the dataset root (`models/` or T-LESS `models_cad/`, `test/` or `test_primesense/`, `test_targets_bop19.json`). Layouts: `popoe.datasets.bop.BOP_LAYOUTS`.

Units: CAD vertices in **mm**; unprojected depth and output `t` in **metres**. BOP CSVs convert `t` back to mm at the edge.

### Detection files

JSONs under `data/detections/` are **not in git**. Directories ship `PROVENANCE.md` + `MANIFEST.sha256`. Place files, then `python scripts/freeze_detections.py --check`.

Hashes and downloads: [CNOS.md](CNOS.md), [NIDS_NET.md](NIDS_NET.md), [SAM6D.md](SAM6D.md), [MUSE.md](MUSE.md), `data/detections/*/PROVENANCE.md`. Official names (`cnos`, `sam6d`, `nids`, `muse`) are reserved; reimplementations write `cnos-lab` / `muse-repro`.

## Three identities

Mixing these is how wrong numbers get cited.

| Identity | What it is | Entry |
|----------|------------|--------|
| **Evaluated default** | Tuned Open3D: `ChampionScorer`, mask floor 100, IoU dedupe 0.9, tau from sampled query extent, ICP on the sparse grid, `--render-backend nvdiffrast`. **Not** a paper-faithful freeze. | `examples/bop_eval.py` with no paper-side flags |
| **Paper-side flags** | Opt-in FreeZe-v2 knobs (`--eq5-terms`, `--tau-diameter`, `--icp-dense`, `--solver gpu-feat`, `--min-mask-pixels 0`, `--mask-iou-dedupe` above 1, …). | `examples/bop_eval.py --help` |
| **Parity oracle** | Byte-identity of historical `RansacSolver` + `FreeZeScorer` vs `examples/freezev2_monolith.py`. Not the BOP loop. | `examples/pipeline_selfcheck.py` |

## Minimal BOP eval

This is the **evaluated default**. A wrong `--bop` root fails rather than writing an all-zero CSV.

```bash
export POPOE_GEDI_PATH=/path/to/gedi
python examples/bop_eval.py \
    --bop /path/to/ycbv \
    --detections data/detections/cnos/cnos-fastsam_ycbv-test.json \
    --out popoe_ycbv.csv \
    --cache /path/to/popoe_cache_ycbv
```

Exactly one of `--detections` or `--sources name=path,...`. Paper-side knobs are on `--help`.

Score with `python -m popoe.metrics.ar` (`POPOE_BOP_TOOLKIT`, `BOP_PATH`; optional `BOP_DATASET`). Same `BOP_LAYOUTS` as `bop_eval.py` / `metrics.vsd` / `metrics.grasp`. Multi-instance CSVs need official bop_toolkit for 1–1 assignment. Submissions need one shared per-image time: `examples/bop_time_normalize.py`.

Solver ranking on GT instances (needs `models_eval/`):

```bash
export POPOE_GEDI_PATH=/path/to/gedi
export POPOE_BOP_TOOLKIT=/path/to/bop_toolkit
python examples/solver_swap_demo.py --bop /path/to/ycbv --obj 5 -n 5 --seed 42
```

## Stages

```python
import popoe  # numpy + scikit-learn

popoe.PoseMethod
popoe.Segmentor, popoe.QueryEncoder, popoe.TargetEncoder
popoe.PoseSolver, popoe.CoarseEstimator, popoe.PoseRefiner, popoe.GeometricRefiner
popoe.PoseScorer, popoe.Selector
popoe.Scene, popoe.ObjectModel, popoe.CanonFrame
popoe.Detection, popoe.PointFeatures, popoe.PoseHypothesis
```

A new method implements `PoseMethod.run`. A new stage drops into `make_correspondence_pipeline` → `Pipeline` or `DirectPoseMethod` — no registry. Missing backends raise `BackendUnavailable`; they never silently substitute. A solver proposes; `ChampionScorer` disposes. Details: [ARCHITECTURE.md](ARCHITECTURE.md).

```python
from popoe import PointFeatures, PoseHypothesis

class MySolver:  # popoe.PoseSolver by structure
    def solve(self, query: PointFeatures, target: PointFeatures
              ) -> list[PoseHypothesis]:
        R, t = my_registration(query.pts, query.feats, target.pts, target.feats)
        return [PoseHypothesis(R=R, t=t, score=..., breakdown={"s_coarse": ...})]
```

## Detections

Evaluated runs consume **precomputed BOP-format files**. CNOS-FastSAM, SAM-6D ISM, and NIDS-Net are named files; union is top-M per source with no cross-source filter.

```python
from popoe.segmentor_detections import BOPDetectionsSegmentor
seg = BOPDetectionsSegmentor(sources={
    "cnos": "data/detections/cnos/cnos-fastsam_ycbv-test.json",
    "nids": "data/detections/nids/nids_wa_sappe_ycbv.json",
}, topk=2)
```

| Source | What | Download |
|--------|------|----------|
| **CNOS-FastSAM** | Official BOP default detections | HuggingFace [`bop-benchmark/bop_extra`](https://huggingface.co/datasets/bop-benchmark/bop_extra) → `cnos-fastsam_{ycbv,lmo}-test.json` |
| **NIDS-Net** | WA_Sappe BOP predictions | UT Dallas Box via [`IRVLUTD/NIDS-Net`](https://github.com/IRVLUTD/NIDS-Net) → `nids_wa_sappe_{ycbv,lmo}.json` |
| **SAM-6D** | BOP method-441 segmentation | `sub_info` → `sam6d_official_{ds}.json`; hashes in `data/detections/sam6d/PROVENANCE.md` |

Producers run in separate environments. Consume files with `BOPDetectionsSegmentor(..., source="cnos"|"nids"|"sam6d")`. `cnos` is official; `cnos-lab` is the local recipe. **MUSE**: paper public, authors' code not; official BOP masks exist; `popoe.segmentor_muse` writes `muse-repro`. How-to: [CNOS.md](CNOS.md), [NIDS_NET.md](NIDS_NET.md), [SAM6D.md](SAM6D.md), [MUSE.md](MUSE.md).

Records are `{scene_id, image_id, category_id, score, segmentation}` (COCO RLE). Real captures may use `mask` / `mask_path`; still 2D, never depth. Frame I/O: `popoe.datasets.frames` (`FrameManifest.depth_scale` is metres/raw). Smoke: `python examples/union_smoke.py --dataset ycbv`. 2D AP (not 6-DoF): `examples/bop_seg_eval.py`.

## Layout

```
src/popoe/               # method-agnostic pipeline
  interfaces.py          # Protocols + data classes
  freeze/                # FreeZe-v2 reference (encoders, fusion, recipes)
examples/  tests/  scripts/  ARCHITECTURE.md
```

## Scripts and examples

Index: [scripts/README.md](scripts/README.md).

| Path | Role |
|------|------|
| `examples/bop_eval.py` | Evaluated BOP loop |
| `examples/bop_time_normalize.py` | Shared per-image time for a submission |
| `examples/bop_seg_eval.py` | COCO mask AP on detections |
| `examples/pipeline_selfcheck.py` | Parity oracle |
| `examples/solver_swap_demo.py` | Solver ranking on GT instances |
| `scripts/freeze_detections.py` | Detection-file identity gate |
| `scripts/make_gt_detections.py` | Oracle `mask_visib` → detections JSON |
| `scripts/check_rerank_symmetry.py` | `--render-rerank` flip-inflation gate |

Ablations (`eval_*_ab.py`, `*_ab_run.sh`, `sar_render_compare.py`, …) live under `scripts/` and are not the eval entry.

## Tests

```bash
pytest tests/    # CPU. CI installs `.[dev,reference]`; GPU / nvdiffrast / GeDi skip.
```

## License

Apache-2.0 ([LICENSE](LICENSE)). Third-party models keep their own licenses ([NOTICE](NOTICE)).
