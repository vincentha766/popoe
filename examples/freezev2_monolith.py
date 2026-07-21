"""The inline FreeZeV2 monolith — demo / parity oracle, not library API.

This is the original single-class pipeline (extract -> RANSAC -> ICP -> score)
kept verbatim so `pipeline_selfcheck.py` can assert the adapter chain
(popoe.freeze.adapters + popoe.adapters) reproduces it bitwise, and so
`solver_swap_demo.py` has a one-object harness. The library equivalents are
`popoe.interfaces.Pipeline` wired via `popoe.freeze.recipes`.
"""

import numpy as np
import torch

from popoe.registration import (
    ransac_pose_estimation, icp_refinement, feature_aware_score, final_score,
)


class FreeZeV2:
    """
    Full FreeZeV2 pipeline for a single object-mask pair.
    Uses /workspace/gedi for geometric features, DINOv2 for visual features.
    DINOv2 and GeDi are shared between query/target extractors to save VRAM.
    """

    def __init__(self, device='cuda', tau_inlier=0.03, tau_icp=0.03,
                 n_ransac=10000, k_corr=10):
        from popoe.freeze.feature_extractor import (
            QueryFeatureExtractor, TargetFeatureExtractor, load_dinov2, load_gedi)
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
