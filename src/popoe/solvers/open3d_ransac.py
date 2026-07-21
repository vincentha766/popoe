"""
A SECOND PoseSolver implementation — proof that the framework is pluggable.

This file adds an alternative to the hand-rolled RANSAC (popoe.adapters.RansacSolver)
WITHOUT touching any other stage: it just implements the `PoseSolver` Protocol
(popoe.interfaces.PoseSolver) and can be dropped into interfaces.Pipeline in
place of RansacSolver. Everything downstream (ICPRefiner, FreeZeScorer, Selector)
is unchanged.

`Open3DFeatureRansacSolver` uses Open3D's correspondence-RANSAC
(`registration_ransac_based_on_feature_matching`, a C++ implementation with edge-
length + distance pruning checkers) on the fused features — a genuinely different
algorithm from the pure-Python `ransac_pose_estimation`. Motivation beyond the
pluggability demo: progress.md finds the -20.8pt YCB-V gap lives mostly in
registration on thin / near-symmetric geometry where RANSAC+ICP struggles;
swapping the solver is the intended lever to attack that (TEASER++ / MAC would be
further PoseSolver implementations added the same way — one new file each).

Like the other solver, this returns ONLY the coarse pose (no ICP, no final
score): ICP is PoseRefiner's job and the s_coarse/s_fine/s_icp combination is
PoseScorer's. s_coarse is computed with the same feature_aware_score the FreeZe
solver uses internally, so the two solvers feed the scorer comparably.
"""

from __future__ import annotations
import numpy as np

from popoe.interfaces import CanonFrame, PointFeatures, PoseHypothesis


class Open3DFeatureRansacSolver:
    """PoseSolver via Open3D feature-matching RANSAC (metre-space points)."""

    def __init__(self, tau_inlier: float = 0.03, ransac_n: int = 3,
                 max_iteration: int = 10000, confidence: float = 0.999,
                 edge_length: float = 0.9, mutual_filter: bool = True,
                 n_restarts: int = 1, subsample: float = 0.7):
        self.tau_inlier = tau_inlier
        self.ransac_n = ransac_n
        self.max_iteration = max_iteration
        self.confidence = confidence
        self.edge_length = edge_length
        self.mutual_filter = mutual_filter
        # n_restarts>1: run RANSAC on `subsample` fractions of the points (varied
        # per restart) and return ALL resulting poses as separate hypotheses.
        # Open3D ranks by geometric fitness and flips on symmetric geometry; by
        # emitting several candidates, the downstream feature-aware PoseScorer +
        # Selector do the disambiguation instead ("geometry proposes, features
        # dispose"). Every hypothesis' s_coarse is still feature_aware_score on
        # the FULL cloud, so candidates are comparable.
        self.n_restarts = n_restarts
        self.subsample = subsample

    def solve(self, query: PointFeatures, target: PointFeatures,
              frame: CanonFrame) -> list[PoseHypothesis]:
        import open3d as o3d
        from popoe.registration import feature_aware_score

        N_q, N_t = len(query.pts), len(target.pts)
        if N_t < self.ransac_n or N_q < self.ransac_n:
            return []

        reg = o3d.pipelines.registration
        dim = query.feats.shape[1]

        def pcd(pts):
            p = o3d.geometry.PointCloud()
            p.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
            return p

        def feat(f, n):
            F = reg.Feature(); F.resize(dim, n); F.data = f.T.astype(np.float64); return F

        def one_run(qp, qf, tp, tf):
            pcd_q, pcd_t = pcd(qp), pcd(tp)
            fq, ft = feat(qf, len(pcd_q.points)), feat(tf, len(pcd_t.points))
            r = reg.registration_ransac_based_on_feature_matching(
                pcd_q, pcd_t, fq, ft,
                mutual_filter=self.mutual_filter,
                max_correspondence_distance=self.tau_inlier,
                estimation_method=reg.TransformationEstimationPointToPoint(False),
                ransac_n=self.ransac_n,
                checkers=[
                    reg.CorrespondenceCheckerBasedOnEdgeLength(self.edge_length),
                    reg.CorrespondenceCheckerBasedOnDistance(self.tau_inlier),
                ],
                criteria=reg.RANSACConvergenceCriteria(
                    max_iteration=self.max_iteration, confidence=self.confidence),
            )
            T = np.asarray(r.transformation)
            return T[:3, :3].copy(), T[:3, 3].copy(), float(r.fitness)

        hyps = []
        for restart in range(max(1, self.n_restarts)):
            if restart == 0:
                qp, qf, tp, tf = query.pts, query.feats, target.pts, target.feats
            else:  # diversify by subsampling (deterministic per restart)
                rng = np.random.default_rng(1000 + restart)
                nq = max(self.ransac_n, int(N_q * self.subsample))
                nt = max(self.ransac_n, int(N_t * self.subsample))
                qi = rng.choice(N_q, size=nq, replace=False)
                ti = rng.choice(N_t, size=nt, replace=False)
                qp, qf, tp, tf = query.pts[qi], query.feats[qi], target.pts[ti], target.feats[ti]
            R, t, fit = one_run(qp, qf, tp, tf)
            # Score every candidate on the FULL cloud so they are comparable.
            s_coarse, _ = feature_aware_score(
                R, t, query.pts, target.pts, query.feats, target.feats, self.tau_inlier)
            hyps.append(PoseHypothesis(R=R, t=t, score=s_coarse,
                                       breakdown={"s_coarse": s_coarse,
                                                  "o3d_fitness": fit, "restart": restart}))
        return hyps
