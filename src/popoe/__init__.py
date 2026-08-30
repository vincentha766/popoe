"""popoe — Pipeline Of Pose Estimation.

A modular 6-DoF object pose framework. See ARCHITECTURE.md.

Importing `popoe` pulls the contract layer (numpy). Heavy modules
(freeze.feature_extractor, registration, renderer, segmentor) import torch /
open3d and must be imported explicitly, e.g.
`from popoe.freeze.recipes import best_encoders`.
"""
from popoe.interfaces import (
    Scene, FrameManifest, ObjectModel, CanonFrame, Detection, PointFeatures,
    PoseHypothesis, Pipeline, Segmentor, PointDescriptor, FeatureFusion,
    QueryEncoder, TargetEncoder, PoseSolver, CoarseEstimator, PoseRefiner,
    PoseScorer, Selector,
)

__version__ = "0.1.0"

__all__ = [
    "Scene", "FrameManifest", "ObjectModel", "CanonFrame", "Detection",
    "PointFeatures", "PoseHypothesis", "Pipeline", "Segmentor",
    "PointDescriptor", "FeatureFusion", "QueryEncoder", "TargetEncoder",
    "PoseSolver", "CoarseEstimator", "PoseRefiner", "PoseScorer", "Selector",
    "__version__",
]
