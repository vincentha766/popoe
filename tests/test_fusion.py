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


def test_w1_extraction_plus_scale_vis_reproduces_any_weight():
    """The contract best_encoders' pin relies on: extracting at vis_weight=1
    and rescaling with recipes.scale_vis(w) must equal extracting directly at
    vis_weight=w. (The old bug: extraction silently happened at the env
    default 0.5, so every 'w' in the sweep was really 0.5*w.)"""
    from popoe.recipes import scale_vis
    rng = np.random.default_rng(3)
    vis = rng.standard_normal((200, 1536)).astype(np.float32)
    geo = rng.standard_normal((200, 64)).astype(np.float32)
    geo[::41] = np.nan
    pca = PCA(n_components=64).fit(vis[~np.isnan(geo).any(1)])

    w1 = DinoGeDiFusion(pca_vis=pca, vis_weight=1.0).fuse(vis, geo)
    for w in (1.0, 0.7, 0.5, 0.3, 0.2):
        direct = DinoGeDiFusion(pca_vis=pca, vis_weight=w).fuse(vis, geo)
        assert np.allclose(scale_vis(w1, w), direct, atol=1e-6), f"w={w}"


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
