"""
FreeZeV2 pose estimation core.
Feature matching + RANSAC + ICP + feature-aware scoring.
"""

import numpy as np
try:  # torch is only needed by the GPU feature path; scoring/RANSAC are numpy
    import torch
    torch.backends.cudnn.enabled = False  # cuDNN init fails on some hosts
except ImportError:
    torch = None
from sklearn.neighbors import KDTree
from scipy.spatial.transform import Rotation
import open3d as o3d


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


def icp_refinement(pts_query, pts_target_dense, R_init, t_init, tau_icp=0.03, max_iter=50):
    """
    ICP refinement (Stage 3, Eq. 6).
    Align pts_query (transformed) to dense target point cloud.
    """
    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector((R_init @ pts_query.T).T + t_init)

    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(pts_target_dense)

    init_T = np.eye(4)
    result = o3d.pipelines.registration.registration_icp(
        src_pcd, tgt_pcd, tau_icp, init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
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


class FreeZeV2:
    """
    Full FreeZeV2 pipeline for a single object-mask pair.
    Uses /workspace/gedi for geometric features, DINOv2 for visual features.
    DINOv2 and GeDi are shared between query/target extractors to save VRAM.
    """

    def __init__(self, device='cuda', tau_inlier=0.03, tau_icp=0.03,
                 n_ransac=10000, k_corr=10):
        from popoe.feature_extractor import QueryFeatureExtractor, TargetFeatureExtractor, load_dinov2, load_gedi
        # Load heavy models once and share
        print("Loading DINOv2...")
        shared_dino = load_dinov2(device)
        print("Loading GeDi...")
        shared_gedi = load_gedi(device)
        self.query_extractor = QueryFeatureExtractor(device, dino=shared_dino, gedi=shared_gedi)
        self.target_extractor = TargetFeatureExtractor(device, dino=shared_dino, gedi=shared_gedi)
        self.tau_inlier = tau_inlier
        self.tau_icp = tau_icp
        self.n_ransac = n_ransac
        self.k_corr = k_corr
        self.device = device

    def precompute_query(self, mesh_path, n_points=5000):
        """Offline: precompute query features from 3D model."""
        import trimesh
        mesh = trimesh.load(mesh_path, force='mesh')
        pts, _ = trimesh.sample.sample_surface_even(mesh, n_points)
        pts = pts / 1000.0  # BOP mesh mm -> m to match depth-unprojected target pcd
        pts = torch.from_numpy(pts.astype(np.float32))

        feats, pts = self.query_extractor.extract_query_features(mesh_path, pts)
        self._pts_query = pts.numpy() if isinstance(pts, torch.Tensor) else pts
        self._feats_query = feats
        return self._pts_query, self._feats_query

    def estimate_pose(self, rgb, depth, mask, intrinsics, pts_target_dense=None):
        """
        Online: estimate 6D pose for a candidate mask.
        Returns T (4x4), score.
        """
        # Share canon_scale from query extractor so GeDi runs on the same
        # canonicalised point cloud scale as the query side.
        self.target_extractor._canon_scale = getattr(
            self.query_extractor, '_canon_scale', 1.0
        )
        pts_sparse, feats_target = self.target_extractor.extract_target_features(
            rgb, depth, mask, intrinsics,
            pca_vis=self.query_extractor._pca_vis
        )

        if pts_sparse is None or len(pts_sparse) < 4:
            return None, 0.0

        # Stage 2: RANSAC feature matching
        R_c, t_c, s_coarse = ransac_pose_estimation(
            self._pts_query, self._feats_query,
            pts_sparse, feats_target,
            n_iters=self.n_ransac,
            tau_inlier=self.tau_inlier,
            k=self.k_corr
        )

        # Stage 3: ICP refinement
        dense_pts = pts_target_dense if pts_target_dense is not None else pts_sparse
        R_f, t_f, s_icp = icp_refinement(self._pts_query, dense_pts, R_c, t_c, self.tau_icp)

        # Re-score at fine level
        s_fine, _ = feature_aware_score(
            R_f, t_f, self._pts_query, pts_sparse,
            self._feats_query, feats_target, self.tau_inlier
        )

        score = final_score(s_coarse, s_fine, s_icp)

        T = np.eye(4)
        T[:3, :3] = R_f
        T[:3, 3] = t_f
        return T, score
