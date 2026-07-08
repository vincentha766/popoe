"""Pose solvers (Stage 2). Each is an independent `PoseSolver` implementation.

Add a new solver as a new module here implementing `.solve(query, target, frame)
-> list[PoseHypothesis]`; nothing else in the pipeline changes.
"""
from popoe.solvers.open3d_ransac import Open3DFeatureRansacSolver

__all__ = ["Open3DFeatureRansacSolver"]
