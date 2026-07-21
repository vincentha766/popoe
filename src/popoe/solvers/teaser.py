"""A FOURTH PoseSolver — TEASER++ certifiable robust registration.

TEASER++ (Yang, Shi & Carlone, T-RO 2021) is the "further PoseSolver
implementation" the Open3D module's docstring anticipated: instead of sampling
minimal triplets (RANSAC's exponential blow-up at high outlier ratios), it
DECOUPLES scale/rotation/translation, prunes correspondences with a pairwise
translation-invariant-measurement max-clique, and solves rotation by GNC-TLS —
robust to >90% outliers, deterministic (no RNG), with optimality certificates.
That failure mode is exactly where progress.md locates the remaining AR gap:
thin / near-symmetric geometry where the correspondence set is outlier-heavy and
RANSAC+ICP converges to the wrong basin.

Like the other solvers this returns ONLY the coarse pose (ICP is PoseRefiner's
job, final scoring is PoseScorer's) in the shared hypothesis shape: `score =
s_coarse` = feature_aware_score on the FULL cloud at w=1, so downstream stages
(ICPRefiner / ChampionScorer / Selector) are unchanged.

Correspondences are built the same way as GPURansacSolver's Eq.3 pool — per-
target top-k query NN by cosine in the w=1 feature space (`meta['feats_w1']`,
falling back to `.feats`) — because TEASER++ is correspondence-based, not a
matcher. Default k=1: the max-clique pruning is O(C^2) in correspondence count,
and TEASER++'s outlier robustness does not need the recall padding that
RANSAC's sampling does.

Requires `teaserpp_python` (built from source — https://github.com/MIT-SPARK/
TEASER-plusplus; no PyPI wheel). The import is deferred to `solve`, so
constructing the solver (e.g. in recipes) never needs the package.

Convention: R maps QUERY -> TARGET (`p_t ≈ R p_q + t`), matching
feature_aware_score and the other solvers.
"""

from __future__ import annotations

import numpy as np

from popoe.interfaces import CanonFrame, PointFeatures, PoseHypothesis


def _import_teaser():
    try:
        import teaserpp_python
    except ImportError as e:
        raise ImportError(
            "TeaserSolver needs teaserpp_python, which has no PyPI wheel. "
            "Build it from source: https://github.com/MIT-SPARK/TEASER-plusplus "
            "(cmake -DBUILD_PYTHON_BINDINGS=ON, then pip install build/python)."
        ) from e
    return teaserpp_python


def _correspondences(pts_q, fq, pts_t, ft, k, mutual_filter, max_corr):
    """Per-target top-k query NN by cosine (GPURansacSolver's Eq.3 pool).
    Returns (src, dst) as (C, 3) arrays, similarity-capped at max_corr."""
    fqn = fq / np.clip(np.linalg.norm(fq, axis=1, keepdims=True), 1e-12, None)
    ftn = ft / np.clip(np.linalg.norm(ft, axis=1, keepdims=True), 1e-12, None)
    sim = ftn @ fqn.T                                  # (Nt, Nq)
    k_eff = min(k, sim.shape[1])
    topi = np.argsort(-sim, axis=1)[:, :k_eff]         # (Nt, k)
    c_t = np.repeat(np.arange(sim.shape[0]), k_eff)
    c_q = topi.reshape(-1)
    c_sim = sim[c_t, c_q]

    if mutual_filter:
        q_best_t = sim.argmax(axis=0)                  # best target per query
        mutual = q_best_t[c_q] == c_t
        if int(mutual.sum()) >= 3:                     # same guard as gpu_ransac
            c_t, c_q, c_sim = c_t[mutual], c_q[mutual], c_sim[mutual]

    if len(c_t) > max_corr:                            # max-clique is O(C^2)
        keep = np.argsort(-c_sim)[:max_corr]
        c_t, c_q = c_t[keep], c_q[keep]
    return pts_q[c_q], pts_t[c_t]


class TeaserSolver:
    """PoseSolver via TEASER++ robust registration (metre-space points).

    `tau_inlier` doubles as TEASER's noise bound, keeping the inlier semantics
    of the RANSAC solvers. Scale estimation is off (query and target are both
    metric). Deterministic — no seed needed."""

    source = "teaser"

    def __init__(self, tau_inlier: float = 0.03, k: int = 1,
                 mutual_filter: bool = False, max_corr: int = 4000,
                 cbar2: float = 1.0, gnc_factor: float = 1.4,
                 rot_max_iterations: int = 100,
                 rot_cost_threshold: float = 1e-12):
        self.tau_inlier = tau_inlier
        self.k = k
        self.mutual_filter = mutual_filter
        self.max_corr = max_corr
        self.cbar2 = cbar2
        self.gnc_factor = gnc_factor
        self.rot_max_iterations = rot_max_iterations
        self.rot_cost_threshold = rot_cost_threshold

    def solve(self, query: PointFeatures, target: PointFeatures,
              frame: CanonFrame) -> list[PoseHypothesis]:
        tpp = _import_teaser()
        from popoe.registration import feature_aware_score

        # w=1 canonical features (A-layer lesson: absolute feature scores are
        # only comparable at w=1); fall back to .feats when meta is absent.
        fq = query.meta.get("feats_w1", query.feats)
        ft = target.meta.get("feats_w1", target.feats)
        if len(query.pts) < 3 or len(target.pts) < 3:
            return []

        src, dst = _correspondences(query.pts, fq, target.pts, ft,
                                    self.k, self.mutual_filter, self.max_corr)
        if len(src) < 3:
            return []

        params = tpp.RobustRegistrationSolver.Params()
        params.cbar2 = self.cbar2
        params.noise_bound = self.tau_inlier
        params.estimate_scaling = False
        params.rotation_estimation_algorithm = (
            tpp.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS)
        params.rotation_gnc_factor = self.gnc_factor
        params.rotation_max_iterations = self.rot_max_iterations
        params.rotation_cost_threshold = self.rot_cost_threshold
        solver = tpp.RobustRegistrationSolver(params)
        # TEASER++ takes (3, C) float64, src -> dst == query -> target.
        solver.solve(np.ascontiguousarray(src.T, dtype=np.float64),
                     np.ascontiguousarray(dst.T, dtype=np.float64))
        sol = solver.getSolution()
        R = np.asarray(sol.rotation, dtype=np.float64)
        t = np.asarray(sol.translation, dtype=np.float64).reshape(3)
        clique = solver.getInlierMaxClique()
        # Failure gate. The 1.1.0 binding's solution has NO `.valid` field, and
        # TEASER can abort internally ("Max clique lower bound equals to zero")
        # yet still hand back an uninitialized solution — which is shape-correct
        # and would flow through ICP/scoring as a real pose. A max clique of <3
        # cannot determine a rotation, and an aborted solve can leave R
        # non-finite / non-orthonormal; treat all of those as "no hypothesis".
        if (not bool(getattr(sol, "valid", True)) or len(clique) < 3
                or not np.isfinite(R).all() or not np.isfinite(t).all()
                or not np.allclose(R @ R.T, np.eye(3), atol=1e-5)
                or np.linalg.det(R) < 0):
            return []

        # s_coarse: same key/shape/space as the other solvers, so ICPRefiner /
        # ChampionScorer are unchanged.
        s_coarse, _ = feature_aware_score(R, t, query.pts, target.pts, fq, ft,
                                          self.tau_inlier)
        return [PoseHypothesis(R=R, t=t, score=s_coarse,
                               breakdown={"s_coarse": s_coarse,
                                          "n_corr": int(len(src)),
                                          "n_clique": len(clique),
                                          "n_rot_inliers": len(solver.getRotationInliers()),
                                          "restart": 0})]
