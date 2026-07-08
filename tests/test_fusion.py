"""Fusion tests — GPU-free (numpy + scikit-learn only)."""
import numpy as np
from sklearn.decomposition import PCA

import popoe
from popoe.fusion import DinoGeDiFusion


def _reference(vis, geo, pca, vis_w):
    """The intended arithmetic, written out independently."""
    valid = ~np.isnan(geo).any(axis=1)
    vis_r = pca.transform(vis)
    l2 = lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    geo_safe = geo.copy(); geo_safe[~valid] = 0
    return np.concatenate([vis_w * l2(vis_r), l2(geo_safe)], axis=1).astype(np.float32)


def test_protocol_conformance():
    assert isinstance(DinoGeDiFusion(), popoe.FeatureFusion)


def test_byte_identity_with_shared_pca():
    rng = np.random.default_rng(0)
    vis = rng.standard_normal((300, 1536)).astype(np.float32)
    geo = rng.standard_normal((300, 64)).astype(np.float32)
    geo[::37] = np.nan
    pca = PCA(n_components=64).fit(vis[~np.isnan(geo).any(1)])
    got = DinoGeDiFusion(pca_vis=pca, vis_weight=0.5).fuse(vis, geo)
    assert np.array_equal(got, _reference(vis, geo, pca, 0.5))


def test_vis_weight_zero_is_pure_geometric():
    rng = np.random.default_rng(1)
    vis = rng.standard_normal((200, 1536)).astype(np.float32)
    geo = rng.standard_normal((200, 64)).astype(np.float32)
    fused = DinoGeDiFusion(vis_weight=0.0).fuse(vis, geo)
    # visual half (first 64 dims) is zeroed out
    assert np.allclose(fused[:, :64], 0.0)
    assert not np.allclose(fused[:, 64:], 0.0)


def test_output_dims():
    rng = np.random.default_rng(2)
    vis = rng.standard_normal((300, 1536)).astype(np.float32)
    geo = rng.standard_normal((300, 32)).astype(np.float32)
    fused = DinoGeDiFusion().fuse(vis, geo)   # vis_dim defaults to geo dim
    assert fused.shape == (300, 64)           # 32 vis + 32 geo
