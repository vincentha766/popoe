# popoe — Pipeline Of Pose Estimation

A modular, **training-free 6-DoF object pose** framework, built for and
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
ObjectModel (CAD) ─┐
                   ├─ Segmentor ─ QueryEncoder ─┐
Scene (RGB-D, K) ──┘            TargetEncoder ──┴─ PoseSolver ─ PoseRefiner* ─ PoseScorer ─ Selector ─ (R, t)
```

The bundled reference implementation reproduces a FreeZe-v2-style pipeline
(DINOv2 visual + GeDi geometric features → RANSAC → ICP → symmetry-aware scoring)
and ships **two** `PoseSolver` implementations to demonstrate pluggability.

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

DINOv2 is pulled via `torch.hub`. See [NOTICE](NOTICE) for upstream licenses —
**each keeps its own license; verify before use.**

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

The two shipped solvers (`popoe.adapters.RansacSolver`,
`popoe.solvers.Open3DFeatureRansacSolver`) are worked examples. A robust backend
like TEASER++ or MAC would be added the same way — one file.

A stage never hides a fallback: if its backend is missing (no package, no
checkpoint, no GPU) it raises `BackendUnavailable` rather than quietly running a
weaker method under the same name. Substitution is the caller's call, and the
caller can see what ran:

```python
from popoe.segmentor import DepthSegmentor, FirstAvailableSegmentor
from popoe.segmentor_cnos import CNOSSegmentor, DepthBoxMasker, DinoWindowSegmentor

seg = FirstAvailableSegmentor([
    CNOSSegmentor(renderer),                                   # SAM2 + DINOv2
    DinoWindowSegmentor(renderer, masker=DepthBoxMasker()),    # no SAM2 needed
    DepthSegmentor(),                                          # no deps at all
])
dets = seg.segment(scene, obj)
seg.last_used      # -> 'cnos' | 'dino-window' | 'depth-cc'
dets[0].source     # per detection; the window segmentor appends its masker,
                   # e.g. 'dino-window+depth-box' — survives into the CSV
```

See [ARCHITECTURE.md](ARCHITECTURE.md#the-availability-contract-no-hidden-fallbacks)
for why (short version: a silent fallback makes results unattributable and
poisons the config-addressed cache).

Because a solver only has to *propose* candidates and the feature-aware
`PoseScorer` + `Selector` *dispose*, `Open3DFeatureRansacSolver(n_restarts=8)`
turns a geometry-only RANSAC (which flips on near-symmetric objects) back to
parity with the feature-aware solver — with no new scoring code. See
[ARCHITECTURE.md](ARCHITECTURE.md#pluggability-proven--a-second-posesolver).

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

**Format notes.** A detections file is a JSON list of records
`{scene_id, image_id, category_id, score, segmentation}` where `segmentation`
is a COCO RLE. The loader (`load_bop_detections`) handles the format variance
seen across these releases without special-casing at the call site:

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
