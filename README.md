# popoe — Pipeline Of Pose Estimation

A modular **6-DoF object pose** framework, built for and
evaluated on the **BOP benchmark**. The pipeline is broken into swappable stages
behind small `Protocol` contracts, so **every step can grow its own method** —
add a segmentor, a feature backbone, a pose solver, a scorer, without touching
the rest.

**Scope**: popoe is the pose library — BOP datasets, metrics, and evaluated
recipes. Downstream applications (grasping, HTTP, a robot stack) live elsewhere
and should call popoe behind the stage contracts, not grow those layers here.

```
ObjectModel (CAD) ─┬─ QueryEncoder ──────────── q, CanonFrame ─┐
                   ├─ Segmentor ─ Detection ─┐                 │
Scene (RGB-D, K) ──┴─────────────────────────┴─ TargetEncoder ─┴─ PoseSolver ─ PoseRefiner* ─ PoseScorer ─ Selector ─ (R, t)
```

The bundled reference implementation reproduces a FreeZe-v2-style pipeline
(DINOv2 visual + GeDi geometric features → RANSAC → ICP → symmetry-aware scoring)
and ships multiple `PoseSolver` implementations to demonstrate pluggability.

> Status: research code, `v0.1`. The framework layer (contracts + fusion) is
> covered by tests and runs on CPU. The FreeZe-style reference implementation
> needs a CUDA GPU, the external models below, a BOP split, and detection
> JSONs (none of those are in this clone). See
> [ARCHITECTURE.md](ARCHITECTURE.md) for the design.

There is **no published AR on this path**. [REPRODUCTION.md](REPRODUCTION.md)
is a placeholder until one frozen recipe is re-run end to end. Do not cite
figures from git history of that file.

### After a clone

| You want | Install | Also needed | Run |
|----------|---------|-------------|-----|
| Read the contracts / write a stage | `pip install -e .` | — | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Check the clone | `pip install -e ".[dev]"` | — | `pytest tests/` (CPU) |
| Run 6-DoF on BOP | `pip install -e ".[reference]"` | CUDA, GeDi, **nvdiffrast**, a BOP split, detection JSONs | [`examples/bop_eval.py`](#minimal-bop-eval) |

`examples/solver_swap_demo.py` is a GPU solver comparison on **ground-truth
instances**. It does not read detection files and is not the BOP eval loop.
The library entry is `PoseMethod.run(scene, obj)`. Build the correspondence
graph with `popoe.freeze.recipes.make_correspondence_pipeline` (returns a
`Pipeline`); `DirectPoseMethod` is the estimator-graph method (e.g. SAM-6D PEM).
`examples/bop_eval.py` writes BOP CSVs (cache, weight sweep, resume,
multi-instance) and scores each encoded pair with `correspond_pair`, the
same kernel `Pipeline.run` uses after encode. `scripts/` holds offline
A/B helpers, not onboarding.

Some tests import OpenCV / pycocotools. Install `.[reference]` as well if
`pytest tests/` should cover those, not only the contract layer.

## Install

```bash
pip install -e .                # framework only (numpy, scikit-learn)
pip install -e ".[reference]"   # + reference impl (torch, open3d, trimesh, opencv, ...)
pip install -e ".[dev]"         # + pytest
```

CPU-only check that the clone is healthy:

```bash
pip install -e ".[dev]"
pytest tests/
```

### External dependencies (not on PyPI)

The reference implementation orchestrates external models/toolkits. Clone them
yourself and **export the env vars**. Unset, they are empty — there is no
lab-host fallback.

| Component | Env var | Where it comes from |
|-----------|---------|---------------------|
| GeDi | `POPOE_GEDI_PATH` | Clone [fabiopoiesi/gedi](https://github.com/fabiopoiesi/gedi). The directory must contain `data/chkpts/3dmatch/chkpt.tar` (see that repo). This is the geometric descriptor, not a popoe sibling. |
| SAM 2 checkpoints | `POPOE_SAM2_CKPT` | Directory holding `sam2.1_hiera_large.pt`. Needed only for live SAM2 / MUSE / CNOS-lab, not for file-backed BOP eval. |
| bop_toolkit | `POPOE_BOP_TOOLKIT` | Clone [thodan/bop_toolkit](https://github.com/thodan/bop_toolkit). Used by `solver_swap_demo.py` and `python -m popoe.metrics.ar`. |
| nvdiffrast | — | GPU CAD renderer. **Required** for the default `bop_eval.py` (`--render-backend nvdiffrast`). Missing it is an error, not a silent trimesh swap. Pass `--render-backend trimesh` or `auto` only if you accept different CAD views (hence different features). |
| Official CNOS / NIDS / SAM-6D producers | `POPOE_CNOS_PATH`, `external/NIDS-Net`, `POPOE_SAM6D_PATH` | Optional. Evaluated BOP runs consume **already-written** detection files; you do not need these producers — or `git submodule update` — unless you want to regenerate detections. |

DINOv2 weights come from `torch.hub` on first encode (`TORCH_HOME` caches them).
See [NOTICE](NOTICE) for upstream licenses — **each keeps its own license;
verify before use.**

Pinned producer source checkouts (optional; skip for file-backed eval):

```bash
git submodule update --init --recursive external/cnos external/NIDS-Net external/SAM-6D
```

A later encode also reads feature knobs from the environment
(`POPOE_QUERY_POINTS`, `POPOE_TARGET_GRID`, `POPOE_DINO_LAYER`,
`POPOE_TWO_SCALE_GEDI`, `POPOE_VIS_DIM`, `POPOE_GEOM_BACKBONE`, …). Unset,
they take the defaults recorded in the eval cache key. Changing one without
a new `--cache` directory reuses stale features.

### BOP data

popoe does not vendor BOP. Download a Classic-Core split from
[bop.felk.cvut.cz/datasets](https://bop.felk.cvut.cz/datasets/) and pass the
dataset root to `--bop` (the directory that contains `models/`, `test/` or
`test_primesense/`, and `test_targets_bop19.json`). Per-dataset layout
quirks (T-LESS/HB primesense split, ITODD `gray/*.tif`, T-LESS `models_cad`)
are in `popoe.datasets.bop.BOP_LAYOUTS`.

Units at the library boundary: CAD mesh vertices in **millimetres**;
unprojected depth and output `t` in **metres**. BOP CSVs convert `t` back to
mm at the edge.

### Detection files

The JSONs under `data/detections/` are **not in git** (large). A clone has
empty directories plus `PROVENANCE.md` and `MANIFEST.sha256`. Place the files
at the paths in the manifest, then:

```bash
python scripts/freeze_detections.py --check
```

Download sources and SHA256s: [CNOS.md](CNOS.md), [NIDS_NET.md](NIDS_NET.md),
[SAM6D.md](SAM6D.md), [MUSE.md](MUSE.md), and
`data/detections/*/PROVENANCE.md`. Official names (`cnos`, `sam6d`, `nids`,
`muse`) are reserved for official artefacts; lab reimplementations write
`cnos-lab` / `muse-repro`.

## Minimal BOP eval

Defaults are the **tuned Open3D** identity (mask pixel floor 100, IoU
dedupe 0.9, tau from the sampled query extent, ICP on the sparse grid
cloud). That is **not** a paper-faithful freeze, and the CSV is **not** a
number to cite.

`--bop` is the dataset root that contains `models/` (T-LESS: `models_cad/`),
`test/` or `test_primesense/`, and `test_targets_bop19.json`. T-LESS/HB use
the primesense split; ITODD uses `gray/*.tif`. Layouts live in
`popoe.datasets.bop.BOP_LAYOUTS`; a wrong root fails rather than writing an
all-zero CSV.

```bash
export POPOE_GEDI_PATH=/path/to/gedi
python examples/bop_eval.py \
    --bop /path/to/ycbv \
    --detections data/detections/cnos/cnos-fastsam_ycbv-test.json \
    --out popoe_ycbv.csv \
    --cache /path/to/popoe_cache_ycbv
```

Pass exactly one of `--detections` (one JSON) or `--sources name=path,...`
(named union; FreeZe-style multi-source). Paper-side knobs (`--eq5-terms`,
`--tau-diameter`, `--icp-dense`, `--solver gpu-feat`, `--render-rerank`,
`--min-mask-pixels 0`, `--mask-iou-dedupe` above 1) are on `--help`. There
is no named recipe that pins them into one command.

Score the CSV with `python -m popoe.metrics.ar` (`POPOE_BOP_TOOLKIT` set).
A BOP submission needs one shared per-image time — run
`examples/bop_time_normalize.py` first.

## Quickstart — the stages

```python
import popoe  # light: only numpy + scikit-learn

# A method is run(scene, obj). Stages are optional building blocks:
popoe.PoseMethod
popoe.Segmentor, popoe.QueryEncoder, popoe.TargetEncoder
popoe.PoseSolver, popoe.CoarseEstimator, popoe.PoseRefiner
popoe.PoseScorer, popoe.Selector

# Data that flows between them:
popoe.Scene, popoe.ObjectModel, popoe.CanonFrame
popoe.Detection, popoe.PointFeatures, popoe.PoseHypothesis
```

A GPU demo that ranks the shipped solvers on the **same encoded pair**
(GT instances, not detection JSONs; needs `models_eval/` as well as `models/`):

```bash
export POPOE_GEDI_PATH=/path/to/gedi
export POPOE_BOP_TOOLKIT=/path/to/bop_toolkit
python examples/solver_swap_demo.py --bop /path/to/ycbv --obj 5 -n 5 --seed 42
```

## Extending — a new method, or a new stage

The library entry is `PoseMethod.run(scene, obj)`. Stages are optional
building blocks; a correspondence method uses encoders + `PoseSolver`, an
estimator method uses `CoarseEstimator`. No base class, no registration.
Example — a new pose solver for the correspondence graph:

```python
# my_solver.py
from popoe import PointFeatures, PoseHypothesis

class MySolver:  # satisfies popoe.PoseSolver structurally
    def solve(self, query: PointFeatures, target: PointFeatures
              ) -> list[PoseHypothesis]:
        R, t = my_registration(query.pts, query.feats, target.pts, target.feats)
        return [PoseHypothesis(R=R, t=t, score=..., breakdown={"s_coarse": ...})]
```

A new **method** implements `PoseMethod.run`. A new **stage** (for example a
solver) drops into the correspondence graph
(`make_correspondence_pipeline` → `Pipeline`) or an estimator graph
(`DirectPoseMethod`) without a registry. The BOP CSV loop is
`examples/bop_eval.py` (see [Minimal BOP eval](#minimal-bop-eval)).
The shipped solvers
(`popoe.solvers.Open3DFeatureRansacSolver`, `popoe.solvers.GPURansacSolver`,
and `popoe.solvers.TeaserSolver`) are worked examples.

A stage never hides a fallback: if its backend is missing (no package, no
checkpoint, no GPU) it raises `BackendUnavailable` rather than quietly running a
weaker method under the same name.

```python
from popoe.segmentor import DepthSegmentor, SAMSegmentor

dets = SAMSegmentor().segment(scene, obj)   # raises if SAM2 is missing
dets[0].source     # 'sam2-amg' — survives into the CSV
```

See [ARCHITECTURE.md](ARCHITECTURE.md#the-availability-contract-no-hidden-fallbacks)
for why (short version: a silent fallback makes results unattributable and
poisons the config-addressed cache).

A solver only has to *propose* candidates; `ChampionScorer` *disposes*. So a
geometry-only RANSAC can emit several hypotheses
(`Open3DFeatureRansacSolver(n_restarts=8)`) and let the existing scorer choose,
with no new scoring code. See
[ARCHITECTURE.md](ARCHITECTURE.md#pluggability-proven--the-posesolver-stage)
for the seam. `examples/solver_swap_demo.py` runs the three shipped solvers
on the same encoded pair.

## Detections (segmentation sources)

The evaluated segmentor consumes **precomputed BOP-format detections**. popoe
reads three open sources — CNOS-FastSAM, SAM-6D ISM, and NIDS-Net — under one
backend and can **union any subset**, reproducing FreeZe-v2's multi-source
segmentation (top-M per source, unioned without cross-source filtering; the
feature-aware scorer disposes). Each is just a named file:

```python
from popoe.segmentor_detections import BOPDetectionsSegmentor
seg = BOPDetectionsSegmentor(sources={          # or a single path=one source
    "cnos": "data/detections/cnos/cnos-fastsam_ycbv-test.json",
    "nids": "data/detections/nids/nids_wa_sappe_ycbv.json",
    # "sam6d": "…",                             # optional third source
}, topk=2)
dets = seg.segment(scene, obj)                  # dets[i].source -> 'cnos'|'nids'|…
```

| Source | What | Download |
|--------|------|----------|
| **CNOS-FastSAM** | Official BOP default detections (FastSAM proposals + DINOv2 re-rank) | HuggingFace [`bop-benchmark/bop_extra`](https://huggingface.co/datasets/bop-benchmark/bop_extra), the default-detections bundle → `cnos-fastsam_{ycbv,lmo}-test.json` |
| **NIDS-Net** | WA_Sappe variant BOP predictions | UT Dallas Box, linked from [`IRVLUTD/NIDS-Net`](https://github.com/IRVLUTD/NIDS-Net) README → "Inference on BOP datasets"; saved as `nids_wa_sappe_{ycbv,lmo}.json` |
| **SAM-6D ISM** | Instance Segmentation Model masks | No public per-dataset file — run [`JiehongLin/SAM-6D`](https://github.com/JiehongLin/SAM-6D) ISM on the BOP test images (GPU); optional |

Official CNOS, NIDS-Net and SAM-6D are intentionally **external producers**,
not popoe dependencies. Their official stacks use heavy and version-pinned
segmentation, foundation-model and pose-estimation packages. The source
checkouts are pinned under `external/`, but runtime should still happen in
separate `uv`/conda projects or services. Export predictions, then consume
them here as named files through `BOPDetectionsSegmentor(..., source="cnos")`
(or `"nids"` / `"sam6d"`).
That keeps `popoe`'s pose backend independent of segmentation-model dependency
conflicts while preserving per-detection source provenance through scoring.
See [CNOS.md](CNOS.md), [NIDS_NET.md](NIDS_NET.md), and [SAM6D.md](SAM6D.md) for
deployment notes and the adapter CLIs.

CNOS naming is deliberately split:

| Source | Meaning |
|--------|---------|
| `cnos` | Official CNOS/CNOS-FastSAM predictions, public BOP files, or `external/cnos` output |
| `cnos-lab` | Local lab recipe (formerly `cnos-v3`): proposal masks -> depth size gate -> DINOv2 foreground-patch rank |

The ensemble's fourth member, **MUSE**, publishes **no code**, so it has no
external producer to adapt — but its **masks are public for all seven
BOP-Classic-Core sets** (BOP `method_info/873`; IDs and SHA256s in
[data/detections/muse/PROVENANCE.md](data/detections/muse/PROVENANCE.md)).
`popoe.segmentor_muse` is a reimplementation from the paper: a live segmentor
that can also dump its masks as a detections JSON, which then unions and
evaluates like any other source. It writes `muse-repro`; the name `muse` stays
reserved for official artefacts. See [MUSE.md](MUSE.md).

```bash
popoe-muse --frame capture/frame_000000.json \
  --classes 9=/templates/obj_000009,14=/templates/obj_000014 \
  --models-info .../models_info.json --out outputs/muse_frame0.json

popoe-bop-muse --bop-root /path/to/ycbv \
  --template-root /templates/ycbv \
  --out outputs/muse-repro_ycbv-test.json \
  --shard-dir outputs/muse-repro_ycbv-test_shards --resume
```

**Format notes.** A detections file is a JSON list of records
`{scene_id, image_id, category_id, score, segmentation}` where `segmentation`
is a COCO RLE. Real-capture files may use a `mask` or `mask_path` alias instead;
these are still 2D masks, never depth. The loader (`load_bop_detections`, alias
`load_detections`) handles the format variance seen across these releases
without special-casing at the call site:

- **Fully-stringified records** — the NIDS WA_Sappe Box release ships every
  field as a string (`"scene_id": "48"`, `"score": "0.74…"`, bbox as a
  stringified list). Coerced at load; a non-integral id is a loud error, not a
  silent truncation.
- **Uncompressed vs compressed RLE** — `counts` may be a run-length **list**
  (uncompressed, both the CNOS and NIDS files here) or a COCO RLE **string**;
  `decode_detection_mask` routes each correctly (a compressed string may itself
  begin with `[`, so the discriminator parses, it does not sniff the first byte).

Files are **not committed** (large; gitignored under `data/detections/`). A
no-GPU end-to-end check over whatever files you have:

```bash
python examples/union_smoke.py --dataset ycbv    # load -> decode -> union -> select
```

To score a segmentation source directly, use COCO mask AP over BOP visible
masks. This evaluates only the 2D proposals, not 6-DoF pose:

```bash
python examples/bop_seg_eval.py \
  --bop /path/to/ycbv \
  --detections data/detections/cnos/cnos-fastsam_ycbv-test.json \
  --targets /path/to/ycbv/test_targets_bop19.json \
  --per-object \
  --out-dir outputs/ycbv_cnos_seg_ap
```

The command builds a merged `gt_coco.json` from `mask_visib`, converts the BOP
detections into `pred_coco.json`, then runs `pycocotools` in `segm` mode and
writes `summary.json`. With `--per-object`, it also writes per-category
`per_object.json` and `per_object.csv` diagnostics for the same single
segmentation source. BOP targets select target images, matching the official
BOP COCO evaluator; wrong-category detections on those images remain false
positives. It needs a complete BOP split with GT masks; sparse local
subsets that only include RGB/depth/poses are useful for pose debugging but
cannot produce segmentation AP.

For real RGB-D captures, keep the same boundary: detections are still **2D**
masks/scores only, and depth stays with the frame. A frame manifest points to
the RGB/depth files, intrinsics, scale, and the per-frame detections file:

```json
{
  "scene_id": 0,
  "image_id": 42,
  "rgb_path": "frames/000042_rgb.png",
  "depth_path": "frames/000042_depth.png",
  "depth_scale": 0.001,
  "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "detections_path": "detections/000042.json"
}
```

The matching detections file can use BOP's `segmentation` field or the local
`mask` alias:

```json
[
  {
    "scene_id": 0,
    "image_id": 42,
    "category_id": 9,
    "score": 0.96,
    "bbox": [120, 80, 230, 210],
    "mask": {"format": "rle", "size": [480, 640], "counts": "..."}
  }
]
```

Load it into the same pipeline types:

```python
from popoe.datasets.frames import load_frame_manifest, load_scene_from_manifest
from popoe.segmentor_detections import BOPDetectionsSegmentor

frame = load_frame_manifest("capture/frame_000042.json")
scene = load_scene_from_manifest(frame)     # depth is now metres
seg = BOPDetectionsSegmentor(frame.detections_path, source="local-cnos")
```

## Layout

```
src/popoe/               # method-agnostic pipeline
  interfaces.py          # stage Protocols + data classes
  registration.py        # RANSAC / ICP / feature-aware scoring primitives
  adapters.py            # ICPRefiner / select_top_instances
  scoring.py             # ChampionScorer (evaluated scorer)
  renderer.py  segmentor.py  segmentor_cnos_lab.py  segmentor_cnos_official.py
  solvers/open3d_ransac.py  solvers/gpu_ransac.py  solvers/teaser.py
  metrics/vsd.py  metrics/ar.py
  datasets/bop.py
  freeze/                # the FreeZe-v2 reference method
    feature_extractor.py # DINOv2 + GeDi encoders
    fusion.py            # FeatureFusion (DinoGeDiFusion)
    adapters.py          # FreeZe encoder/scorer stage adapters
    recipes.py           # default (tuned Open3D) stage configuration
examples/  tests/  ARCHITECTURE.md
```

## Tests

```bash
pytest            # framework-layer tests (no GPU): fusion byte-identity, Protocol wiring
```

## License

Apache-2.0 (see [LICENSE](LICENSE)). Third-party models keep their own licenses
(see [NOTICE](NOTICE)).
