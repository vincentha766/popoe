# popoe — Pipeline Of Pose Estimation

A modular **6-DoF object pose** framework, evaluated on **BOP**. Stages sit behind small `Protocol` contracts so a segmentor, backbone, solver, or scorer can grow without rewriting the rest.

```
ObjectModel (CAD) ─┬─ QueryEncoder ──────────── q, CanonFrame ─┐
                   ├─ Segmentor ─ Detection ─┐                 │
Scene (RGB-D, K) ──┴─────────────────────────┴─ TargetEncoder ─┴─ PoseSolver ─ PoseRefiner* ─ PoseScorer ─ Selector ─ (R, t)
```

The reference method is FreeZe-v2-style (DINOv2 + GeDi → RANSAC → ICP → symmetry-aware scoring) with several `PoseSolver` implementations.

**Scope**: a pose library — BOP datasets, metrics, evaluated recipes. Grasping, HTTP, and robot stacks belong elsewhere and should call these contracts.

> Research code, `v0.1`. Contracts and fusion are CPU-tested. The reference run needs CUDA, GeDi, nvdiffrast, a BOP split, and detection JSONs; none of those ship in a clone.

**Docs**: [ARCHITECTURE.md](ARCHITECTURE.md) (seams and stage protocols) · [docs/invariants.md](docs/invariants.md) (guards that must stay) · [docs/sources/](docs/sources/README.md) (external detection and pose producers) · [scripts/README.md](scripts/README.md) (diagnostics and ablations)

## Quickstart

| You want | Install | Also needed | Then |
|----------|---------|-------------|------|
| Read or write a stage | `pip install -e .` | — | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Check the clone | `pip install -e ".[dev]"` | — | `pytest tests/` |
| Run 6-DoF on BOP | `pip install -e ".[reference]"` | CUDA, GeDi, nvdiffrast, a BOP split, detection JSONs | [Running a BOP eval](#running-a-bop-eval) |

```bash
pip install -e .                # numpy + scikit-learn
pip install -e ".[reference]"   # torch, open3d, trimesh, opencv, …
pip install -e ".[dev]"         # + pytest
pytest tests/                   # CPU; GPU / nvdiffrast / GeDi / OpenCV / Open3D skip when missing
```

CI installs `.[dev,reference]` and runs the same CPU suite.

`PoseMethod.run(scene, obj)` is the library entry. See [Writing a stage](#writing-a-stage) for the contracts, and [ARCHITECTURE.md](ARCHITECTURE.md) for how the two method graphs are composed.

## Install

### External dependencies (not on PyPI)

Clone these yourself and export the env vars. Unset means empty — there is no implicit host path.

| Component | Env var | Notes |
|-----------|---------|-------|
| GeDi | `POPOE_GEDI_PATH` | [fabiopoiesi/gedi](https://github.com/fabiopoiesi/gedi) with `data/chkpts/3dmatch/chkpt.tar` |
| nvdiffrast | — | **Required** for the default `bop_eval.py` and `best_encoders()`. Missing it is an error, not a silent trimesh swap. `--render-backend trimesh` or `auto` is opt-in and gives different CAD views |
| bop_toolkit | `POPOE_BOP_TOOLKIT` | [thodan/bop_toolkit](https://github.com/thodan/bop_toolkit) — needed by `solver_swap_demo.py` and `python -m popoe.metrics.ar` |
| SAM 2 | `POPOE_SAM2_CKPT` | Directory holding `sam2.1_hiera_large.pt`. Live SAM2 / MUSE / CNOS-lab only |
| CNOS / NIDS / SAM-6D producers | `POPOE_CNOS_PATH`, `external/NIDS-Net`, `POPOE_SAM6D_PATH` | Optional — evaluated runs consume already-written files. See [docs/sources/](docs/sources/README.md) |

DINOv2 comes from `torch.hub` (`TORCH_HOME`). Licences: [NOTICE](NOTICE) — each upstream keeps its own, so verify before use.

Producer checkouts are optional and only needed to regenerate artefacts:

```bash
git submodule update --init --recursive external/cnos external/NIDS-Net external/SAM-6D
```

### BOP data

Download a Classic-Core split from [bop.felk.cvut.cz/datasets](https://bop.felk.cvut.cz/datasets/). `--bop` is the dataset root, holding `models/` (or T-LESS `models_cad/`), `test/` (or `test_primesense/`), and `test_targets_bop19.json`. Per-dataset layouts are in `popoe.datasets.bop.BOP_LAYOUTS`.

Units: CAD vertices in **mm**; unprojected depth and output `t` in **metres**. BOP CSVs convert `t` back to mm at the edge.

### Detection files

The JSONs under `data/detections/` are not in git. Each directory ships a `PROVENANCE.md` with downloads and SHA256s, plus a `MANIFEST.sha256`. Place the files, then verify:

```bash
python scripts/freeze_detections.py --check
```

Which producer writes which file, and what each source tag means: [docs/sources/](docs/sources/README.md).

### Encoder knobs

`POPOE_QUERY_POINTS`, `POPOE_TARGET_GRID`, `POPOE_DINO_LAYER`, `POPOE_TWO_SCALE_GEDI`, `POPOE_VIS_DIM`, `POPOE_GEOM_BACKBONE` and friends are recorded in the eval cache key. Changing one without a new `--cache` replays stale features — see [docs/invariants.md](docs/invariants.md#cache-keys-fingerprint-config-and-content).

## Three identities

Mixing these up is how wrong numbers get cited.

| Identity | What it is | Entry |
|----------|------------|--------|
| **Evaluated default** | Tuned Open3D: `ChampionScorer`, mask floor 100, IoU dedupe 0.9, tau from sampled query extent, ICP on the sparse grid, `--render-backend nvdiffrast`. Not a paper-faithful freeze. | `examples/bop_eval.py` with no paper-side flags |
| **Paper-side flags** | Opt-in FreeZe-v2 knobs: `--eq5-terms`, `--tau-diameter`, `--icp-dense`, `--solver gpu-feat`, `--min-mask-pixels 0`, `--mask-iou-dedupe` above 1, … | `examples/bop_eval.py --help` |
| **Parity oracle** | Byte-identity of the historical `RansacSolver` + `FreeZeScorer` against `examples/freezev2_monolith.py`. Not the BOP loop. | `examples/pipeline_selfcheck.py` |

There are no frozen headline BOP numbers in this repository. Do not cite internal or unpublished runs as popoe results.

## Running a BOP eval

This is the **evaluated default**. A wrong `--bop` root fails rather than writing an all-zero CSV.

```bash
export POPOE_GEDI_PATH=/path/to/gedi
python examples/bop_eval.py \
    --bop /path/to/ycbv \
    --detections data/detections/cnos/cnos-fastsam_ycbv-test.json \
    --out popoe_ycbv.csv \
    --cache /path/to/popoe_cache_ycbv
```

Pass exactly one of `--detections` or `--sources name=path,...`. Paper-side knobs are on `--help`.

Score the CSV:

```bash
python -m popoe.metrics.ar      # needs POPOE_BOP_TOOLKIT, BOP_PATH; optional BOP_DATASET
```

`metrics.ar`, `metrics.vsd`, `metrics.grasp` and `bop_eval.py` all share `BOP_LAYOUTS`. The local AR/VSD helpers score one row per target and hard-fail on multi-instance CSVs; 1–1 assignment for those needs the official bop_toolkit. A submission also needs one shared per-image time, which `examples/bop_time_normalize.py` writes.

Solver ranking on GT instances (needs `models_eval/`) — not detections, and not the BOP loop:

```bash
export POPOE_GEDI_PATH=/path/to/gedi
export POPOE_BOP_TOOLKIT=/path/to/bop_toolkit
python examples/solver_swap_demo.py --bop /path/to/ycbv --obj 5 -n 5 --seed 42
```

## Using detections

Evaluated runs consume **precomputed BOP-format files**. CNOS-FastSAM, SAM-6D ISM, NIDS-Net and MUSE are named files behind one loader; a union keeps top-M per source with no cross-source filter.

```python
from popoe.segmentor_detections import BOPDetectionsSegmentor

seg = BOPDetectionsSegmentor(sources={
    "cnos": "data/detections/cnos/cnos-fastsam_ycbv-test.json",
    "nids": "data/detections/nids/nids_wa_sappe_ycbv.json",
}, topk=2)
```

Records are `{scene_id, image_id, category_id, score, segmentation}` with COCO RLE masks; real captures may use `mask` or `mask_path` instead. They are always 2D — depth travels with the frame, via `popoe.datasets.frames` (`FrameManifest.depth_scale` is metres per raw unit).

Official source tags (`cnos`, `sam6d`, `nids`, `muse`) are reserved for the original authors' artefacts; reimplementations write `cnos-lab` / `muse-repro`. Downloads, per-producer setup and the full tag table: [docs/sources/](docs/sources/README.md).

Two quick checks:

```bash
python examples/union_smoke.py --dataset ycbv    # multi-source union loads
python examples/bop_seg_eval.py                  # 2D mask AP, not 6-DoF
```

## Writing a stage

```python
import popoe  # numpy + scikit-learn

popoe.PoseMethod
popoe.Segmentor, popoe.QueryEncoder, popoe.TargetEncoder
popoe.PoseSolver, popoe.CoarseEstimator, popoe.PoseRefiner, popoe.GeometricRefiner
popoe.PoseScorer, popoe.Selector
popoe.Scene, popoe.ObjectModel, popoe.CanonFrame
popoe.Detection, popoe.PointFeatures, popoe.PoseHypothesis
```

A new method implements `PoseMethod.run`. A new stage drops into `make_correspondence_pipeline` → `Pipeline`, or into `DirectPoseMethod` — there is no registry. Conformance is structural: match the signature, skip the base class.

```python
from popoe import PointFeatures, PoseHypothesis

class MySolver:  # popoe.PoseSolver by structure
    def solve(self, query: PointFeatures, target: PointFeatures
              ) -> list[PoseHypothesis]:
        R, t = my_registration(query.pts, query.feats, target.pts, target.feats)
        return [PoseHypothesis(R=R, t=t, score=..., breakdown={"s_coarse": ...})]
```

Two rules a new stage inherits: a missing backend raises `BackendUnavailable` instead of substituting a weaker method, and a solver only proposes — `ChampionScorer` disposes. Both are spelled out in [docs/invariants.md](docs/invariants.md).

## Layout

```
src/popoe/               # method-agnostic pipeline
  interfaces.py          # Protocols + data classes
  freeze/                # FreeZe-v2 reference (encoders, fusion, recipes)
examples/                # runnable entry points (table below)
scripts/                 # diagnostics and ablations — scripts/README.md
tests/                   # CPU, GPU-free
docs/                    # invariants, external sources
```

| `examples/` | Role |
|------|------|
| `bop_eval.py` | The evaluated BOP loop |
| `bop_time_normalize.py` | Shared per-image time for a submission |
| `bop_seg_eval.py` | COCO mask AP on detections |
| `union_smoke.py` | Multi-source detection union smoke test |
| `pipeline_selfcheck.py` | Parity oracle |
| `freezev2_monolith.py` | Reference monolith the oracle compares against |
| `solver_swap_demo.py` | Solver ranking on GT instances |
| `rule_replay.py` | Replay arbitration rules over a `--cand-csv` dump |

Ablation drivers and diagnostics live under `scripts/` and are indexed in [scripts/README.md](scripts/README.md). None of them is the eval entry.

## License

Apache-2.0 ([LICENSE](LICENSE)). Third-party models keep their own licences ([NOTICE](NOTICE)).
