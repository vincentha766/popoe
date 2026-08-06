# popoe — Pipeline Of Pose Estimation

A modular **6-DoF object pose** framework, built for and
evaluated on the **BOP benchmark**. The pipeline is broken into swappable stages
behind small `Protocol` contracts, so **every step can grow its own method** —
add a segmentor, a feature backbone, a pose solver, a scorer, without touching
the rest.

**Scope**: popoe owns benchmark-grade pose estimation (BOP datasets, metrics,
evaluated-best recipes). Applications — robot grasping, AR, inspection — live in
their own repositories and consume popoe as a dependency behind the
`PoseEstimator`-style seam (see e.g. a lab grasping stack wiring
`popoe.freeze.recipes` into its own pipeline).

```
ObjectModel (CAD) ─┬─ QueryEncoder ──────────── q, CanonFrame ─┐
                   ├─ Segmentor ─ Detection ─┐                 │
Scene (RGB-D, K) ──┴─────────────────────────┴─ TargetEncoder ─┴─ PoseSolver ─ PoseRefiner* ─ PoseScorer ─ Selector ─ (R, t)
```

The bundled reference implementation reproduces a FreeZe-v2-style pipeline
(DINOv2 visual + GeDi geometric features → RANSAC → ICP → symmetry-aware scoring)
and ships multiple `PoseSolver` implementations to demonstrate pluggability.

> Status: research code, `v0.1`. The framework layer (contracts + fusion) is
> covered by tests; the reference implementation runs on a CUDA GPU with the
> external models below. See [ARCHITECTURE.md](ARCHITECTURE.md) for the design
> and the verification story.

## Install

```bash
pip install -e .                # framework only (numpy, scikit-learn)
pip install -e ".[reference]"   # + reference impl (torch, open3d, trimesh, opencv, ...)
pip install -e ".[dev]"         # + pytest
```

### External dependencies (not on PyPI)

The reference implementation orchestrates external models/toolkits — install
these separately and point popoe at them via env vars:

| Component | Env var | Notes |
|-----------|---------|-------|
| GeDi checkpoint + repo | `POPOE_GEDI_PATH` (default `/workspace/gedi`) | geometric descriptor |
| SAM 2 checkpoints | `POPOE_SAM2_CKPT` (default `/workspace/sam2_checkpoints`) | segmentation |
| bop_toolkit | `POPOE_BOP_TOOLKIT` (default `/workspace/bop_toolkit`) | metrics (VSD/MSSD/MSPD) |
| nvdiffrast | — | optional, falls back to trimesh CPU rendering |
| Official CNOS producer | `POPOE_CNOS_PATH` or `external/cnos` submodule | optional detections source |
| NIDS-Net producer | `external/NIDS-Net` submodule | optional detections source |
| SAM-6D producer | `POPOE_SAM6D_PATH` or `external/SAM-6D` submodule | optional ISM detections / PEM pose source |

DINOv2 is pulled via `torch.hub`. See [NOTICE](NOTICE) for upstream licenses —
**each keeps its own license; verify before use.**

For pinned producer source checkouts:

```bash
git submodule update --init --recursive external/cnos external/NIDS-Net external/SAM-6D
```

## Quickstart — the stages

```python
import popoe  # light: only numpy + scikit-learn

# The contracts (Protocols) any implementation satisfies:
popoe.Segmentor, popoe.QueryEncoder, popoe.TargetEncoder
popoe.PoseSolver, popoe.PoseRefiner, popoe.PoseScorer, popoe.Selector

# Data that flows between them:
popoe.Scene, popoe.ObjectModel, popoe.CanonFrame
popoe.Detection, popoe.PointFeatures, popoe.PoseHypothesis
```

Run the reference pipeline (needs a CUDA GPU + the external deps + a BOP dataset):

```bash
# Adapter pipeline is bitwise-identical to the inline reference (acceptance check):
python examples/pipeline_selfcheck.py --bop /path/to/ycbv --obj 5 -n 3

# Swap the pose solver with one line and compare vs GT:
python examples/solver_swap_demo.py  --bop /path/to/ycbv --obj 5 -n 5
```

## Extending — add your own method for a step

Each stage is a `Protocol`: implement the method, drop it in. No base class, no
registration. Example — a new pose solver is one new file:

```python
# my_solver.py
from popoe import PointFeatures, CanonFrame, PoseHypothesis

class MySolver:  # satisfies popoe.PoseSolver structurally
    def solve(self, query: PointFeatures, target: PointFeatures,
              frame: CanonFrame) -> list[PoseHypothesis]:
        R, t = my_registration(query.pts, query.feats, target.pts, target.feats)
        return [PoseHypothesis(R=R, t=t, score=..., breakdown={"s_coarse": ...})]
```

```python
from popoe import Pipeline
pipe = Pipeline(segmentor=..., query_encoder=..., target_encoder=...,
                solver=MySolver(), refiners=[...], selector=..., scorer=...)
best = pipe.run(scene, obj)
```

The shipped solvers (`popoe.adapters.RansacSolver`,
`popoe.solvers.Open3DFeatureRansacSolver`, `popoe.solvers.GPURansacSolver`,
and `popoe.solvers.TeaserSolver`) are worked examples. Another robust backend
like MAC would be added the same way — one file.

A stage never hides a fallback: if its backend is missing (no package, no
checkpoint, no GPU) it raises `BackendUnavailable` rather than quietly running a
weaker method under the same name. Substitution is the caller's call, and the
caller can see what ran:

```python
from popoe.segmentor import DepthSegmentor, FirstAvailableSegmentor
from popoe.segmentor_cnos import CNOSSegmentor, DepthBoxMasker, DinoWindowSegmentor

seg = FirstAvailableSegmentor([
    CNOSSegmentor(renderer),                                   # SAM2 + DINOv2, source=cnos-live
    DinoWindowSegmentor(renderer, masker=DepthBoxMasker()),    # no SAM2 needed
    DepthSegmentor(),                                          # no deps at all
])
dets = seg.segment(scene, obj)
seg.last_used      # -> 'cnos-live' | 'dino-window' | 'depth-cc'
dets[0].source     # per detection; the window segmentor appends its masker,
                   # e.g. 'dino-window+depth-box' — survives into the CSV
```

See [ARCHITECTURE.md](ARCHITECTURE.md#the-availability-contract-no-hidden-fallbacks)
for why (short version: a silent fallback makes results unattributable and
poisons the config-addressed cache).

A solver only has to *propose* candidates; the feature-aware `PoseScorer` +
`Selector` *dispose*. So a geometry-only RANSAC can emit several hypotheses
(`Open3DFeatureRansacSolver(n_restarts=8)`) and let the existing scorer choose,
with no new scoring code. Measured on MSSD over YCB-V obj 5, that roughly halves
the 1-shot median error — though it does not catch the hand-rolled feature-aware
solver. See
[ARCHITECTURE.md](ARCHITECTURE.md#pluggability-proven--the-posesolver-stage).

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
them here as named files or through the provenance-specific wrappers
`popoe.segmentor_cnos_official.CNOSDetectionsSegmentor`,
`popoe.segmentor_nids.NIDSNetDetectionsSegmentor`, and
`popoe.segmentor_sam6d.SAM6DIsmDetectionsSegmentor`.
That keeps `popoe`'s pose backend independent of segmentation-model dependency
conflicts while preserving per-detection source provenance through scoring.
See [CNOS.md](CNOS.md), [NIDS_NET.md](NIDS_NET.md), and [SAM6D.md](SAM6D.md) for
deployment notes and the adapter CLIs.

CNOS naming is deliberately split:

| Source | Meaning |
|--------|---------|
| `cnos` | Official CNOS/CNOS-FastSAM predictions, public BOP files, or `external/cnos` output |
| `cnos-lab` | Local lab recipe (formerly `cnos-v3`): proposal masks -> depth size gate -> DINOv2 foreground-patch rank |
| `cnos-live` | Existing simplified live CNOS-style segmentor (`CNOSSegmentor`), not an official result |

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
  interfaces.py          # stage Protocols + data classes + reference Pipeline
  registration.py        # RANSAC / ICP / feature-aware scoring primitives
  adapters.py            # generic stage adapters (RansacSolver/ICPRefiner/selector)
  scoring.py             # ChampionScorer (evaluated scorer)
  renderer.py  segmentor.py  segmentor_cnos.py  visualizer.py
  solvers/open3d_ransac.py  solvers/gpu_ransac.py  solvers/teaser.py
  metrics/vsd.py  metrics/ar.py
  datasets/bop.py
  freeze/                # the FreeZe-v2 reference method
    feature_extractor.py # DINOv2 + GeDi encoders
    fusion.py            # FeatureFusion (DinoGeDiFusion)
    adapters.py          # FreeZe encoder/scorer stage adapters
    recipes.py           # evaluated-best configuration
examples/  tests/  ARCHITECTURE.md
```

## Tests

```bash
pytest            # framework-layer tests (no GPU): fusion byte-identity, Protocol wiring
```

## License

Apache-2.0 (see [LICENSE](LICENSE)). Third-party models keep their own licenses
(see [NOTICE](NOTICE)).
