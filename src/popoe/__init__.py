"""popoe — Pipeline Of Pose Estimation.

A modular, training-free 6-DoF object pose framework: every stage
(segmentation, feature encoding, fusion, pose solving, refinement, scoring,
selection) is a swappable component behind a Protocol, so each step can grow its
own method. See ARCHITECTURE.md.

The top-level package holds the method-agnostic pieces (interfaces, solvers,
segmentors, registration primitives, cache, metrics); the FreeZe-v2 reference
method (DINOv2+GeDi encoders, fusion, scorer, evaluated recipes) lives in the
`popoe.freeze` subpackage.

Importing `popoe` only pulls the lightweight contract + fusion layer (numpy /
scikit-learn). The heavy implementation modules (freeze.feature_extractor,
registration, renderer, segmentor, ...) import torch / open3d and are imported
explicitly, e.g. `from popoe.freeze.recipes import best_encoders`.
"""
from popoe.interfaces import (
    Scene, ObjectModel, CanonFrame, Detection, PointFeatures, PoseHypothesis,
    Pipeline, Segmentor, FeatureFusion, QueryEncoder, TargetEncoder,
    PoseSolver, PoseRefiner, PoseScorer, Selector, Metric,
)
from popoe.freeze.fusion import DinoGeDiFusion

__version__ = "0.1.0"

__all__ = [
    "Scene", "ObjectModel", "CanonFrame", "Detection", "PointFeatures",
    "PoseHypothesis", "Pipeline", "Segmentor", "FeatureFusion", "QueryEncoder",
    "TargetEncoder", "PoseSolver", "PoseRefiner", "PoseScorer", "Selector",
    "Metric", "DinoGeDiFusion", "__version__",
]
