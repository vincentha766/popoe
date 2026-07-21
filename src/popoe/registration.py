"""
popoe.registration — method-agnostic registration and scoring primitives.

Feature matching (Eq. 3), feature-aware RANSAC (Stage 2), ICP refinement
(Stage 3, Eq. 6) and the score combination (Eq. 5/7). These are the numeric
building blocks the solver / refiner / scorer stages wrap; they carry no
FreeZe-specific state (encoders, fusion, PCA) and depend only on
numpy / scikit-learn (open3d is imported lazily, inside icp_refinement only).

Historically these lived in `popoe.pose_estimator` next to the inline FreeZeV2
monolith; they were extracted so that generic stages (solvers/, scoring,
adapters) no longer import a FreeZe-named module. `popoe.pose_estimator`
remains as a deprecated re-export shim.
"""

import numpy as np
from sklearn.neighbors import KDTree


def top_k_correspondences(pts_target, feats_target, feats_query, k=10):
    """
    Sparse-to-dense feature matching (Eq. 3).
    For each target point, find k nearest query points in fused feature space.

    pts_target: (N_T, 3)
    feats_target: (N_T, D)  sparse target features
    feats_query: (N_Q, D)   dense query features
    Returns: list of (target_idx, [query_idx_k]) pairs
    """
    # Cosine similarity -> use L2 on normalised vectors
    ft_norm = feats_target / (np.linalg.norm(feats_target, axis=1, keepdims=True) + 1e-8)
    fq_norm = feats_query / (np.linalg.norm(feats_query, axis=1, keepdims=True) + 1e-8)

    tree = KDTree(fq_norm)
    dists, indices = tree.query(ft_norm, k=k)
    return indices  # (N_T, k)


def _svd_pose(src, dst):
    """Estimate rigid transform from matched point pairs via SVD."""
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = mu_d - R @ mu_s
    return R, t


def _geometric_prune(pts_q, pts_t, q_idxs, t_idxs, thr=0.05):
    """
    Prune triplet correspondences where geometry is inconsistent.
    Keep triplets where relative edge lengths roughly match.
    """
    n = len(q_idxs)
    if n < 3:
        return list(range(n))

    kept = []
    for i in range(n):
        for j in range(i+1, n):
            dq = np.linalg.norm(pts_q[q_idxs[i]] - pts_q[q_idxs[j]])
            dt = np.linalg.norm(pts_t[t_idxs[i]] - pts_t[t_idxs[j]])
            if dq < 1e-6 or dt < 1e-6:
                continue
            ratio = dq / dt
            if 0.5 < ratio < 2.0:
                kept.extend([i, j])
    return list(set(kept)) if kept else list(range(min(3, n)))


def feature_aware_score(R, t, pts_query, pts_target, feats_query, feats_target, tau_inlier):
    """
    Eq. (5): S_feat^coarse = (1/|P_T^sparse|) * sum cos(f_T^j, f_Q^i)
    for inlier set I where ||R*p_Q^i + t - p_T^j|| < tau_inlier.
    """
    transformed_q = (R @ pts_query.T).T + t  # (N_Q, 3)
    tree = KDTree(transformed_q)
    dists, nn_idx = tree.query(pts_target, k=1)
    dists = dists[:, 0]
    nn_idx = nn_idx[:, 0]

    inliers = dists < tau_inlier
    if inliers.sum() == 0:
        return 0.0, inliers

    ft = feats_target[inliers]
    fq = feats_query[nn_idx[inliers]]
    ft_n = ft / (np.linalg.norm(ft, axis=1, keepdims=True) + 1e-8)
    fq_n = fq / (np.linalg.norm(fq, axis=1, keepdims=True) + 1e-8)
    cos_sims = (ft_n * fq_n).sum(axis=1)
    score = cos_sims.mean()
    return float(score), inliers


def ransac_pose_estimation(pts_query, feats_query, pts_target, feats_target,
                           n_iters=10000, tau_inlier=0.03, k=10):
    """
    RANSAC with feature-aware scoring (Stage 2).
    Returns best (R, t, score).
    """
    nn_indices = top_k_correspondences(pts_target, feats_target, feats_query, k=k)
    N_T = len(pts_target)

    best_score = -1
    best_R, best_t = np.eye(3), np.zeros(3)

    rng = np.random.default_rng(42)

    for _ in range(n_iters):
        # Sample 3 target points
        tidxs = rng.choice(N_T, size=3, replace=False)

        # For each, pick one of its k query neighbours
        qidxs = np.array([rng.choice(nn_indices[ti]) for ti in tidxs])

        src = pts_query[qidxs]
        dst = pts_target[tidxs]

        # Geometric pruning: check relative distances consistent
        dq = np.linalg.norm(src[0] - src[1])
        dt = np.linalg.norm(dst[0] - dst[1])
        if dt < 1e-6 or not (0.3 < dq/dt < 3.0):
            continue

        try:
            R, t = _svd_pose(src, dst)
        except Exception:
            continue

        score, _ = feature_aware_score(R, t, pts_query, pts_target, feats_query, feats_target, tau_inlier)

        if score > best_score:
            best_score = score
            best_R, best_t = R, t

    return best_R, best_t, best_score


def icp_refinement(pts_query, pts_target_dense, R_init, t_init, tau_icp=0.03,
                   max_iter=2000, rel_tol=1e-6):
    """
    ICP refinement (Stage 3, Eq. 6).
    Align pts_query (transformed) to dense target point cloud.
    """
    import open3d as o3d   # lazy: the only open3d user in this module, so the
    # rest (matching / RANSAC / scoring, numpy-only) works without the
    # `reference` extra installed
    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector((R_init @ pts_query.T).T + t_init)

    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(pts_target_dense)

    init_T = np.eye(4)
    result = o3d.pipelines.registration.registration_icp(
        src_pcd, tgt_pcd, tau_icp, init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(rel_tol, rel_tol, max_iter)
    )

    delta = result.transformation
    R_delta = delta[:3, :3]
    t_delta = delta[:3, 3]

    R_fine = R_delta @ R_init
    t_fine = R_delta @ t_init + t_delta
    icp_inlier_ratio = result.fitness

    return R_fine, t_fine, icp_inlier_ratio


def final_score(s_coarse, s_fine, s_icp, alpha=1.0, beta=1.0, gamma=1.0):
    """Eq. (7): S_final = S_feat^coarse^alpha * S_feat^fine^beta * S_ICP^gamma"""
    s_c = max(s_coarse, 0)
    s_f = max(s_fine, 0)
    s_i = max(s_icp, 0)
    return (s_c ** alpha) * (s_f ** beta) * (s_i ** gamma)
