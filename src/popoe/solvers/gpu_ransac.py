"""A THIRD PoseSolver — gedi's vectorized GPU RANSAC, ported.

Motivation (B layer): Open3D's C++ correspondence-RANSAC ranks hypotheses by
GEOMETRIC inlier count and cannot take a custom fitness, so the feature
similarity can only re-rank the survivors (the A layer). To put feature
agreement INSIDE hypothesis selection — changing which hypotheses survive, not
just their order — we need our own RANSAC. This is gedi's batched implementation
(`freezev2_sweep_vis_weight.gpu_ransac`): vectorised triplet sampling + edge-
length pruning + batched Kabsch/SVD, ~54 ms for 10k hypotheses on a GPU. It runs
on CPU too (small scales), so the whole thing is unit-testable without a GPU.

This module ports the SELECTION only (RANSAC). ICP refinement and final scoring
stay the existing stages (ICPRefiner / ChampionScorer), exactly as for
Open3DFeatureRansacSolver — GPURansacSolver returns the same shape (a list of
coarse `PoseHypothesis`, `score = s_coarse`, breakdown carrying `s_coarse`).

`fitness`:
  * ``"geometric"`` (default) — rank hypotheses by inlier COUNT, a faithful port
    of a correspondence-RANSAC (verifiable against Open3D alone). Zero-perturbs
    the mainline: the default solver stays Open3D.
  * ``"feature"`` — the B-layer increment: rank by the paper's Eq.5 feature-
    aware score, ``Σ_inlier cos(f_q, f_t) / |P_T|``. The denominator is the
    FIXED sparse-target count |P_T|, never the inlier count — normalise-by-
    inlier (mean cosine) lets a few high-similarity spurious correspondences
    beat many true ones (ch3 tax #2: -31pt, AR 0.37). Puts feature agreement
    INSIDE hypothesis selection (which hypotheses survive), not just the A-layer
    re-ranking of survivors.

Convention: R maps QUERY -> TARGET (`p_t ≈ R p_q + t`), matching
feature_aware_score and Open3DFeatureRansacSolver.
"""

from __future__ import annotations

import numpy as np

from popoe.interfaces import CanonFrame, PointFeatures, PoseHypothesis

_EDGE_RATIO = 0.9          # Open3D CorrespondenceCheckerBasedOnEdgeLength(0.9)


def _gpu_ransac(pts_q, feats_q, pts_t, feats_t, thr, iters, k, min_inliers,
                fitness, mutual_filter, device, seed):
    """Batched RANSAC. Returns (R, t, fitness_value, n_inliers) as numpy/floats,
    or None if degenerate. `feats_*` are the (already chosen) w=1 features.

    Faithful to gedi's gpu_ransac; the only additions are an explicit torch
    Generator (determinism for tests) and the selectable fitness."""
    import torch

    if iters < 1:
        return None
    pq = torch.as_tensor(np.ascontiguousarray(pts_q), dtype=torch.float32, device=device)
    pt = torch.as_tensor(np.ascontiguousarray(pts_t), dtype=torch.float32, device=device)
    fq = torch.nn.functional.normalize(
        torch.as_tensor(np.ascontiguousarray(feats_q), dtype=torch.float32, device=device), dim=1)
    ft = torch.nn.functional.normalize(
        torch.as_tensor(np.ascontiguousarray(feats_t), dtype=torch.float32, device=device), dim=1)
    N_t = pt.shape[0]
    if N_t < 3 or pq.shape[0] < 3:
        return None

    # Eq.3: top-k query NNs per target point by cosine similarity.
    sim = ft @ fq.T                                   # (Nt, Nq)
    k_eff = min(k, sim.shape[1])
    topv, topi = sim.topk(k_eff, dim=1)               # (Nt, k) cosines + indices
    c_t = torch.arange(N_t, device=device).repeat_interleave(k_eff)
    c_q = topi.reshape(-1)
    c_sim = topv.reshape(-1)                          # (C,) cosine of each corr

    # mutual filter restricts the SAMPLING pool (fewer spurious triplets);
    # scoring still uses the full top-k pool. OFF by default, matching the gedi
    # reference (which only enables it under FREEZEV2_GPU_MUTUAL=1).
    if mutual_filter:
        q_best_t = sim.argmax(dim=0)                  # best target per query
        mutual = q_best_t[c_q] == c_t
        m_idx = (mutual.nonzero(as_tuple=True)[0] if int(mutual.sum()) >= 3
                 else torch.arange(c_t.shape[0], device=device))
    else:
        m_idx = torch.arange(c_t.shape[0], device=device)
    if c_t.shape[0] < 3:
        return None

    gen = torch.Generator(device=device).manual_seed(int(seed))
    sample = torch.randint(0, m_idx.shape[0], (iters, 3), generator=gen, device=device)
    idx = m_idx[sample]
    Pq = pq[c_q[idx]]                                 # (B, 3, 3)
    Pt = pt[c_t[idx]]

    # Edge-length consistency (Open3D checker semantics, ratio > 0.9).
    pairs = [(0, 1), (1, 2), (0, 2)]
    eq = torch.stack([(Pq[:, a] - Pq[:, b]).norm(dim=1) for a, b in pairs], 1)
    et = torch.stack([(Pt[:, a] - Pt[:, b]).norm(dim=1) for a, b in pairs], 1)
    lo = torch.minimum(eq, et); hi = torch.maximum(eq, et)
    valid = (hi > 1e-6).all(1) & ((lo / hi.clamp_min(1e-12)) > _EDGE_RATIO).all(1)

    # Batched Kabsch: R maps query -> target.
    qm = Pq.mean(1, keepdim=True); tm = Pt.mean(1, keepdim=True)
    H = (Pq - qm).transpose(1, 2) @ (Pt - tm)
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.transpose(1, 2)
    det = torch.linalg.det(V @ U.transpose(1, 2))
    D = torch.eye(3, device=device).expand(iters, 3, 3).clone()
    D[:, 2, 2] = det
    R = V @ D @ U.transpose(1, 2)                     # (B, 3, 3)
    t = tm.squeeze(1) - (R @ qm.transpose(1, 2)).squeeze(2)

    # Score every hypothesis over the FULL correspondence pool.
    src, dst = pq[c_q], pt[c_t]                       # (C, 3)
    best = torch.full((iters,), -1e9, device=device)
    n_in_all = torch.zeros(iters, dtype=torch.long, device=device)
    CH = 2048
    for s0 in range(0, iters, CH):
        s1 = min(s0 + CH, iters)
        moved = torch.einsum("bij,cj->bci", R[s0:s1], src) + t[s0:s1, None, :]
        d = (moved - dst[None]).norm(dim=2)           # (b, C)
        inl = d < thr
        n_in = inl.sum(1)
        if fitness == "feature":
            # Eq.5: Σ_inlier cos(f_q, f_t) / |P_T^sparse| — the denominator is
            # the FIXED sparse-target count, NEVER the inlier count. Dividing by
            # n_in (mean cosine) lets a tiny set of high-similarity spurious
            # correspondences outscore a large set of true ones; that exact bug
            # (ch3 reproduction tax #2) collapsed real-data AR to 0.37 (-31pt).
            # Fixed |P_T| makes the score reward inlier QUANTITY x quality.
            val = (c_sim[None] * inl).sum(1) / float(N_t)
        else:  # geometric: inlier fraction (argmax == inlier count)
            val = n_in.to(torch.float32) / float(N_t)
        score = torch.where(n_in >= min_inliers, val, torch.full_like(val, -1e9))
        best[s0:s1] = score
        n_in_all[s0:s1] = n_in
    best = torch.where(valid, best, torch.full_like(best, -1e9))

    b = int(best.argmax())
    if best[b] < -1e8:
        return None
    return (R[b].cpu().numpy().astype(np.float64),
            t[b].cpu().numpy().astype(np.float64),
            float(best[b]), int(n_in_all[b]))


class GPURansacSolver:
    """PoseSolver via gedi's batched GPU RANSAC (metre-space points, w=1 feats).

    Args mirror Open3DFeatureRansacSolver where they overlap. `fitness` selects
    the hypothesis-ranking score ('geometric' | 'feature'); `device=None` picks
    CUDA when available else CPU. Deterministic given `seed`."""

    source = "gpu-ransac"

    def __init__(self, tau_inlier: float = 0.03, iters: int = 10000,
                 k: int = 10, min_inliers: int = 6,
                 fitness: str = "geometric", mutual_filter: bool = False,
                 device: str | None = None, seed: int = 42):
        if fitness not in ("geometric", "feature"):
            raise ValueError(
                f"fitness must be 'geometric' or 'feature', got {fitness!r}")
        self.tau_inlier = tau_inlier
        self.iters = iters
        self.k = k
        self.min_inliers = min_inliers
        self.fitness = fitness
        self.mutual_filter = mutual_filter
        self.device = device
        self.seed = seed

    def solve(self, query: PointFeatures, target: PointFeatures,
              frame: CanonFrame) -> list[PoseHypothesis]:
        import torch
        from popoe.pose_estimator import feature_aware_score

        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        # w=1 canonical features (A-layer lesson: absolute feature scores are
        # only comparable at w=1); fall back to .feats when meta is absent.
        fq = query.meta.get("feats_w1", query.feats)
        ft = target.meta.get("feats_w1", target.feats)

        out = _gpu_ransac(query.pts, fq, target.pts, ft, thr=self.tau_inlier,
                          iters=self.iters, k=self.k, min_inliers=self.min_inliers,
                          fitness=self.fitness, mutual_filter=self.mutual_filter,
                          device=dev, seed=self.seed)
        if out is None:
            return []
        R, t, fit, n_in = out
        # s_coarse is the provisional A-layer signal: feature_aware_score on the
        # FULL cloud (w=1) — the SAME mean-cosine key/shape the Open3D solver
        # emits, so ICPRefiner/ChampionScorer are unchanged. IMPORTANT: it is
        # mean-over-inliers, NOT the B-layer selection fitness; do not read it as
        # the Eq.5 score (that would reintroduce the mean-cosine hijack this
        # solver's fitness avoids). The fixed-|P_T| Eq.5 score that actually
        # RANKED hypotheses is `gpu_score`.
        s_coarse, _ = feature_aware_score(R, t, query.pts, target.pts, fq, ft,
                                          self.tau_inlier)
        # gpu_score: the winning hypothesis's internal ranking score in the
        # chosen mode (feature: Σcos/|P_T|; geometric: inlier-correspondences/
        # |P_T|, which can exceed 1). NOT Open3D's fitness — do not compare to
        # o3d_fitness. n_inliers is the raw inlier-correspondence count.
        return [PoseHypothesis(R=R, t=t, score=s_coarse,
                               breakdown={"s_coarse": s_coarse,
                                          "gpu_score": fit, "n_inliers": n_in,
                                          "fitness_mode": self.fitness,
                                          "restart": 0})]
