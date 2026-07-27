"""Fusion tests — GPU-free (numpy + scikit-learn only)."""
import numpy as np
import pytest
from sklearn.decomposition import PCA

import popoe
from popoe.freeze.fusion import DinoGeDiFusion, IdentityReduction


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
    from popoe.freeze.recipes import scale_vis
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


# ── Reduction is PCA projection or nothing ──────────────────────────────

def test_degenerate_cloud_raises_instead_of_truncating():
    """Regression: with too few valid geometric rows to fit a PCA, fuse() used
    to fall back to `vis_feats[:, :vis_dim]` — the raw first 64 DINO dims
    standing in for a PCA projection, under the same name and with nothing in
    the cache key to show for it."""
    rng = np.random.default_rng(17)
    vis = rng.standard_normal((100, 1536)).astype(np.float32)
    geo = rng.standard_normal((100, 64)).astype(np.float32)
    geo[10:] = np.nan                      # 10 valid rows, vis_dim is 64

    with pytest.raises(ValueError, match="no PCA could be fitted"):
        DinoGeDiFusion(vis_weight=1.0).fuse(vis, geo)


def test_mismatched_pca_width_raises():
    """A PCA fitted on different-width features is the wrong basis, not a
    reason to silently slice."""
    rng = np.random.default_rng(19)
    pca = PCA(n_components=4).fit(rng.standard_normal((80, 16)))
    vis = rng.standard_normal((50, 1536)).astype(np.float32)
    geo = rng.standard_normal((50, 64)).astype(np.float32)

    with pytest.raises(ValueError, match="fitted on 16-D"):
        DinoGeDiFusion(pca_vis=pca, vis_weight=1.0).fuse(vis, geo)


def test_no_reduction_needed_is_identity_not_substitution():
    """The one legitimate non-PCA path: the visual branch already has the
    target width, so passing it through reduces nothing and substitutes
    nothing. Must NOT raise, and must not invent a PCA."""
    rng = np.random.default_rng(23)
    vis = rng.standard_normal((50, 64)).astype(np.float32)
    geo = rng.standard_normal((50, 64)).astype(np.float32)

    fu = DinoGeDiFusion(vis_weight=1.0)
    fused = fu.fuse(vis, geo)

    # RECORDED as identity, not left as None: None means "missing basis" and is
    # refused at the install boundary, so an identity query must be able to say
    # so. (Regression: leaving it None broke POPOE_VIS_DIM=n_vis end to end.)
    assert fu.pca_vis == IdentityReduction()
    assert fu.pca_vis is not None
    assert fused.shape == (50, 128)
    l2 = lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    assert np.allclose(fused[:, :64], l2(vis), atol=1e-6)


def test_identity_query_survives_the_cache_sidecar_and_installs():
    """End-to-end for the no-reduction config (POPOE_VIS_DIM == n_vis): the
    query records identity, it pickles through the cache sidecar as bop_eval
    stores it, and install_pca accepts it — where a None would be refused."""
    import pickle
    from popoe.freeze.adapters import FreeZeTargetEncoder

    rng = np.random.default_rng(29)
    vis_q = rng.standard_normal((300, 64)).astype(np.float32)
    geo_q = rng.standard_normal((300, 64)).astype(np.float32)

    q_fusion = DinoGeDiFusion(vis_weight=1.0)
    q_fusion.fuse(vis_q, geo_q)
    snapshot = pickle.loads(pickle.dumps(q_fusion.pca_vis))   # via the .pkl
    assert snapshot == IdentityReduction(), "compared by type, not identity"

    enc = FreeZeTargetEncoder(_FusionOnlyExtractor())
    enc.install_pca(snapshot)                                  # must not raise

    vis_t = rng.standard_normal((80, 64)).astype(np.float32)
    geo_t = rng.standard_normal((80, 64)).astype(np.float32)
    fused = enc.ex.fusion.fuse(vis_t, geo_t)

    l2 = lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    assert np.allclose(fused[:, :64], l2(vis_t), atol=1e-6)


def test_identity_marker_rejects_a_width_disagreement():
    """Identity is only valid while the two sides agree on the visual width."""
    rng = np.random.default_rng(31)
    fu = DinoGeDiFusion(vis_weight=1.0, pca_vis=IdentityReduction())
    with pytest.raises(ValueError, match="identity reduction was recorded"):
        fu.fuse(rng.standard_normal((40, 128)).astype(np.float32),
                rng.standard_normal((40, 64)).astype(np.float32))


# ── The PCA basis must never be silently re-fitted ──────────────────────

class _FusionOnlyExtractor:
    """Stand-in for TargetFeatureExtractor: install_pca touches only .fusion."""
    def __init__(self):
        self.fusion = DinoGeDiFusion(vis_weight=1.0)


def test_pca_none_means_FIT_not_skip():
    """The premise behind install_pca's guard.

    `pca_vis is None` reads like "no projection to apply", but fuse() treats it
    as "fit one here" — and a target cloud has plenty of valid points to
    succeed with. So a None snapshot on the target side does not degrade
    gracefully; it manufactures a second, unrelated basis.
    """
    rng = np.random.default_rng(7)
    vis = rng.standard_normal((900, 1536)).astype(np.float32)
    geo = rng.standard_normal((900, 64)).astype(np.float32)

    fu = DinoGeDiFusion(vis_weight=1.0)
    assert fu.pca_vis is None
    fu.fuse(vis, geo)
    assert fu.pca_vis is not None, "fuse() fitted a PCA from the data it was given"


def test_install_pca_refuses_none():
    """Regression: install_pca(None) used to hand the target side a blank
    fusion, which then fitted its OWN basis from target features while the
    cached query features stayed in the query basis — cosines compared across
    two unrelated bases, silently. The live trigger is a query cache entry
    whose PCA sidecar is missing (examples/bop_eval.py)."""
    from popoe.freeze.adapters import FreeZeTargetEncoder

    rng = np.random.default_rng(11)
    pca = PCA(n_components=4).fit(rng.standard_normal((80, 16)))
    enc = FreeZeTargetEncoder(_FusionOnlyExtractor())

    enc.install_pca(pca)
    assert enc.ex.fusion.pca_vis is pca

    with pytest.raises(ValueError, match="install_pca"):
        enc.install_pca(None)
    # and it must not have half-applied the bad install
    assert enc.ex.fusion.pca_vis is pca


def test_installed_pca_is_reused_verbatim_on_the_target_side():
    """The positive half: a real snapshot is reused, never re-fitted."""
    from popoe.freeze.adapters import FreeZeTargetEncoder

    rng = np.random.default_rng(13)
    vis_q = rng.standard_normal((3000, 1536)).astype(np.float32)
    geo_q = rng.standard_normal((3000, 64)).astype(np.float32)
    vis_t = (rng.standard_normal((900, 1536)) + 0.4).astype(np.float32)
    geo_t = rng.standard_normal((900, 64)).astype(np.float32)

    q_fusion = DinoGeDiFusion(vis_weight=1.0)
    q_fusion.fuse(vis_q, geo_q)
    snapshot = q_fusion.pca_vis

    enc = FreeZeTargetEncoder(_FusionOnlyExtractor())
    enc.install_pca(snapshot)
    got = enc.ex.fusion.fuse(vis_t, geo_t)

    assert enc.ex.fusion.pca_vis is snapshot, "the basis was replaced"
    assert np.array_equal(got, DinoGeDiFusion(pca_vis=snapshot,
                                              vis_weight=1.0).fuse(vis_t, geo_t))


def test_output_dims():
    rng = np.random.default_rng(2)
    vis = rng.standard_normal((300, 1536)).astype(np.float32)
    geo = rng.standard_normal((300, 32)).astype(np.float32)
    fused = DinoGeDiFusion().fuse(vis, geo)   # vis_dim defaults to geo dim
    assert fused.shape == (300, 64)           # 32 vis + 32 geo
