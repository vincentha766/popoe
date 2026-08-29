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
from typing import Optional

import numpy as np

from popoe.interfaces import CanonFrame, PointFeatures, PoseHypothesis


class Open3DFeatureRansacSolver:
    """PoseSolver via Open3D feature-matching RANSAC (metre-space points)."""

    def __init__(self, tau_inlier: float = 0.03, ransac_n: int = 3,
                 max_iteration: int = 10000, confidence: float = 0.999,
                 edge_length: float = 0.9, mutual_filter: bool = True,
                 n_restarts: int = 1, subsample: float = 0.7,
                 seed: Optional[int] = None, corr_topk: int = 0):
        self.tau_inlier = tau_inlier
        self.ransac_n = ransac_n
        self.max_iteration = max_iteration
        self.confidence = confidence
        self.edge_length = edge_length
        self.mutual_filter = mutual_filter
        # n_restarts>1: run RANSAC on `subsample` fractions of the points (varied
        # per restart) and return ALL resulting poses as separate hypotheses.
        # Open3D ranks by geometric fitness alone, so emitting several candidates
        # hands the choice to the downstream feature-aware PoseScorer + Selector
        # instead. Every hypothesis' s_coarse is still feature_aware_score on the
        # FULL cloud, so candidates are comparable.
        # Measured on MSSD over YCB-V obj 5 (REPRODUCTION.md Solver A/B ledger):
        # this roughly halves median MSSD vs n_restarts=1 and wins head-to-head
        # 72:33, but does NOT reach the hand-rolled feature-aware RansacSolver.
        # An older claim of parity was withdrawn (ISSUES.md 2026-07-26); cite the
        # MSSD numbers, never a raw rotation angle.
        self.n_restarts = n_restarts
        self.subsample = subsample
        # Open3D's RANSAC draws from a GLOBAL RNG that it does not seed itself,
        # and 0.17's registration_ransac_based_on_feature_matching takes no seed
        # argument — so back-to-back identical calls return different poses. The
        # subsampling below is already deterministic (default_rng(1000+restart));
        # this is the other half. `seed=None` keeps the historical unseeded
        # behaviour so the evaluated mainline is not silently perturbed; set it
        # for anything whose numbers get cited (`bop_eval --seed`).
        # It changes RESULTS but not FEATURES, so — unlike the encoder knobs —
        # it deliberately stays OUT of the feature cache key: the cache stores
        # query/target features, which no solver touches, and keying it there
        # would invalidate every existing cache for nothing. It belongs in the
        # run's provenance instead, which is where bop_eval prints it.
        self.seed = seed
        # corr_topk > 0 switches correspondence construction from Open3D's
        # internal matching (kNN in feature space + optional mutual filter,
        # i.e. k=1) to a precomputed top-k set: for each TARGET point, its k
        # best QUERY matches by feature similarity — the convention
        # gpu_ransac.py already uses. FreeZeV2 Sec. IV-A: "For correspondence
        # estimation, we use a top-k strategy with k = 10." 0 keeps the
        # historical k=1 path byte-identical; the mutual filter does not apply
        # to the precomputed set (the paper has no mutual condition).
        self.corr_topk = corr_topk

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

        def one_run(qp, qf, tp, tf, restart):
            # Per-restart, not per-solve: a restart's pose then does not depend
            # on how many restarts ran before it.
            if self.seed is not None:
                o3d.utility.random.seed(self.seed + restart)
            pcd_q, pcd_t = pcd(qp), pcd(tp)
            common = dict(
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
            if self.corr_topk > 0:
                # (Nt, Nq) similarity; rows of the fused feature space have
                # constant norm (two unit-norm halves), but normalise anyway so
                # the ordering is cosine by construction, not by accident.
                qn = qf / np.maximum(np.linalg.norm(qf, axis=1, keepdims=True), 1e-12)
                tn = tf / np.maximum(np.linalg.norm(tf, axis=1, keepdims=True), 1e-12)
                sim = tn @ qn.T
                k_eff = min(self.corr_topk, sim.shape[1])
                top = np.argpartition(-sim, k_eff - 1, axis=1)[:, :k_eff]
                t_idx = np.repeat(np.arange(sim.shape[0]), k_eff)
                pairs = np.stack([top.reshape(-1), t_idx], axis=1)  # [query, target]
                r = reg.registration_ransac_based_on_correspondence(
                    pcd_q, pcd_t,
                    o3d.utility.Vector2iVector(pairs.astype(np.int32)),
                    **common)
            else:
                fq = feat(qf, len(pcd_q.points))
                ft = feat(tf, len(pcd_t.points))
                r = reg.registration_ransac_based_on_feature_matching(
                    pcd_q, pcd_t, fq, ft,
                    mutual_filter=self.mutual_filter, **common)
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
            R, t, fit = one_run(qp, qf, tp, tf, restart)
            # Score every candidate on the FULL cloud so they are comparable.
            s_coarse, _ = feature_aware_score(
                R, t, query.pts, target.pts, query.feats, target.feats, self.tau_inlier)
            hyps.append(PoseHypothesis(R=R, t=t, score=s_coarse,
                                       breakdown={"s_coarse": s_coarse,
                                                  "o3d_fitness": fit, "restart": restart}))
        return hyps
