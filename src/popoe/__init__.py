"""popoe — Pipeline Of Pose Estimation.

A modular, training-free 6-DoF object pose framework: every stage
(segmentation, feature encoding, fusion, pose solving, refinement, scoring,
selection) is a swappable component behind a Protocol, so each step can grow its
own method. See ARCHITECTURE.md.

Importing `popoe` only pulls the lightweight contract + fusion layer (numpy /
scikit-learn). The reference implementation modules (feature_extractor,
pose_estimator, renderer, segmentor, ...) import torch / open3d and are imported
explicitly, e.g. `from popoe.pose_estimator import FreeZeV2`.
"""
from popoe.interfaces import (
    Scene, ObjectModel, CanonFrame, Detection, PointFeatures, PoseHypothesis,
    Pipeline, Segmentor, FeatureFusion, QueryEncoder, TargetEncoder,
    PoseSolver, PoseRefiner, PoseScorer, Selector, Metric,
)
from popoe.fusion import DinoGeDiFusion

__version__ = "0.1.0"

__all__ = [
    "Scene", "ObjectModel", "CanonFrame", "Detection", "PointFeatures",
    "PoseHypothesis", "Pipeline", "Segmentor", "FeatureFusion", "QueryEncoder",
    "TargetEncoder", "PoseSolver", "PoseRefiner", "PoseScorer", "Selector",
    "Metric", "DinoGeDiFusion", "__version__",
]
