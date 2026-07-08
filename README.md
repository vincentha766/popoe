# popoe — Pipeline Of Pose Estimation

A modular, **training-free 6-DoF object pose** framework. The pipeline is broken
into swappable stages behind small `Protocol` contracts, so **every step can grow
its own method** — add a segmentor, a feature backbone, a pose solver, a scorer,
without touching the rest.

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

Because a solver only has to *propose* candidates and the feature-aware
`PoseScorer` + `Selector` *dispose*, `Open3DFeatureRansacSolver(n_restarts=8)`
turns a geometry-only RANSAC (which flips on near-symmetric objects) back to
parity with the feature-aware solver — with no new scoring code. See
[ARCHITECTURE.md](ARCHITECTURE.md#pluggability-proven--a-second-posesolver).

## Layout

```
src/popoe/
  interfaces.py        # stage Protocols + data classes + reference Pipeline
  fusion.py            # FeatureFusion (DinoGeDiFusion)
  adapters.py          # reference stage implementations (FreeZe encoders/solver/refiner/scorer)
  feature_extractor.py # DINOv2 + GeDi encoders
  pose_estimator.py    # RANSAC / ICP / feature-aware scoring
  renderer.py  segmentor.py  segmentor_cnos.py  visualizer.py
  solvers/open3d_ransac.py
  metrics/vsd.py  metrics/ar.py
  datasets/bop.py
examples/  tests/  ARCHITECTURE.md
```

## Tests

```bash
pytest            # framework-layer tests (no GPU): fusion byte-identity, Protocol wiring
```

## License

Apache-2.0 (see [LICENSE](LICENSE)). Third-party models keep their own licenses
(see [NOTICE](NOTICE)).
