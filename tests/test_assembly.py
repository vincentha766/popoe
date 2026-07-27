"""popoe.assembly — the sharing seams and the pool that feeds them.

Everything here is CPU-only: heavy models are stood in for by sentinels via
the pool's ``_load_*`` seams, and the sam2 package by fake modules where the
lazy ``_load`` path itself is under test.
"""
import sys
import types

import pytest

from popoe.assembly import (ModelPool, cnos_lab_segmentor_from_pool,
                            muse_segmentor_from_pool)
from popoe.segmentor import SAMSegmentor
from popoe.segmentor_cnos_lab import (DinoV2ForegroundPatchExtractor,
                                     SAM2AMGMaskProposer)
from popoe.interfaces import ObjectModel
from popoe.segmentor_muse import (DinoV2ClsGemEmbedder, MuseClass,
                                  SAM2BoxMaskRefiner, build_muse_segmentor)


class _Sentinel:
    """Stands in for a loaded model; anything attribute-shaped would do."""


def _two_classes():
    # MuseSegmentor's relative score is a softmax ACROSS classes and
    # refuses a single-class registry.
    return [MuseClass(obj=ObjectModel(obj_id=i, mesh_path=f"obj_{i:06d}.ply",
                                      diameter=0.1),
                      template_dir=f"templates/obj_{i:06d}")
            for i in (1, 2)]


# ── ModelPool ───────────────────────────────────────────────────────────

def test_pool_getter_caches_single_instance():
    pool = ModelPool(device="cpu")
    calls = []

    def load():
        calls.append(1)
        return _Sentinel()

    pool._load_dinov2 = load
    first = pool.dinov2()
    assert pool.dinov2() is first
    assert len(calls) == 1


def test_pool_getters_are_independent():
    pool = ModelPool(device="cpu")
    d, s = _Sentinel(), _Sentinel()
    pool._load_dinov2 = lambda: d
    pool._load_sam2 = lambda: s
    assert pool.dinov2() is d
    assert pool.sam2() is s


# ── injection seams on the DINOv2 components ────────────────────────────

def test_embedder_uses_injected_model_without_loading():
    m = _Sentinel()
    emb = DinoV2ClsGemEmbedder(device="cpu", model=m)
    assert emb.model is m


def test_patch_extractor_uses_injected_model_without_loading():
    m = _Sentinel()
    ex = DinoV2ForegroundPatchExtractor(device="cpu", model=m)
    assert ex.model is m


def test_injection_does_not_change_config_identity():
    plain = DinoV2ClsGemEmbedder(device="cpu").config()
    injected = DinoV2ClsGemEmbedder(device="cpu", model=_Sentinel()).config()
    assert plain == injected

    plain_seg = SAMSegmentor(device="cpu")
    injected_seg = SAMSegmentor(device="cpu", sam_model=_Sentinel())
    assert plain_seg.model_size == injected_seg.model_size


# ── injection seams on the SAM2 components ──────────────────────────────

@pytest.fixture
def fake_sam2(monkeypatch):
    """Install fake sam2 modules so _load can run without the package.

    build_sam2_model would need a checkpoint on disk, so these tests double as
    proof that an injected model BYPASSES it: reaching build_sam2_model at all
    would fail loudly.
    """
    class FakePredictor:
        def __init__(self, model):
            self.model = model

    class FakeAMG:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs

    root = types.ModuleType("sam2")
    pred_mod = types.ModuleType("sam2.sam2_image_predictor")
    pred_mod.SAM2ImagePredictor = FakePredictor
    amg_mod = types.ModuleType("sam2.automatic_mask_generator")
    amg_mod.SAM2AutomaticMaskGenerator = FakeAMG
    monkeypatch.setitem(sys.modules, "sam2", root)
    monkeypatch.setitem(sys.modules, "sam2.sam2_image_predictor", pred_mod)
    monkeypatch.setitem(sys.modules, "sam2.automatic_mask_generator", amg_mod)


def test_box_refiner_load_wraps_injected_model(fake_sam2):
    m = _Sentinel()
    predictor = SAM2BoxMaskRefiner("cpu", sam_model=m)._load()
    assert predictor.model is m


def test_amg_proposer_load_wraps_injected_model(fake_sam2):
    m = _Sentinel()
    generator = SAM2AMGMaskProposer(device="cpu", sam_model=m)._load()
    assert generator.model is m


def test_sam_segmentor_load_wraps_injected_model(fake_sam2):
    m = _Sentinel()
    generator = SAMSegmentor(device="cpu", sam_model=m)._load()
    assert generator.model is m


# ── wiring helpers ──────────────────────────────────────────────────────

def _stub_pool():
    pool = ModelPool(device="cpu")
    pool._load_dinov2 = lambda: _Sentinel()
    pool._load_sam2 = lambda: _Sentinel()
    return pool


def test_muse_from_pool_shares_pool_models():
    pool = _stub_pool()
    seg = muse_segmentor_from_pool(pool, _two_classes())
    assert seg.embedder._model is pool.dinov2()
    assert seg.refiner._sam_model is pool.sam2()
    # the template bank must embed with the SAME embedder the proposals use,
    # or template and proposal features come from different G3 knobs
    assert seg.template_bank.embedder is seg.embedder


def test_cnos_lab_from_pool_shares_pool_models():
    pool = _stub_pool()
    seg = cnos_lab_segmentor_from_pool(pool, "templates/")
    assert seg.template_bank.extractor._model is pool.dinov2()
    assert seg.proposer._sam_model is pool.sam2()
    # the proposer must describe the checkpoint it was actually handed
    assert seg.proposer.model_size == pool.sam2_size


def test_build_muse_segmentor_accepts_prebuilt_components():
    emb = DinoV2ClsGemEmbedder(device="cpu", model=_Sentinel())
    ref = SAM2BoxMaskRefiner("cpu", sam_model=_Sentinel())
    seg = build_muse_segmentor(_two_classes(), embedder=emb, refiner=ref)
    assert seg.embedder is emb
    assert seg.refiner is ref
    assert seg.template_bank.embedder is emb


def test_pose_encoders_refuse_non_freeze_backbone():
    from popoe.assembly import pose_encoders_from_pool
    pool = ModelPool(device="cpu", dinov2_model="dinov2_vitl14")
    with pytest.raises(ValueError):
        pose_encoders_from_pool(pool)


# ── the forward path with an injected model ─────────────────────────────
# _mean/_std moved out of the loader property into _ensure_norm_stats(); these
# pin that every forward path still builds them when the loader never ran.

class _FakeDino:
    """forward_features-shaped stand-in; deterministic, CPU, tiny."""

    def __init__(self, n_tokens: int, dim: int = 8):
        self.n_tokens = n_tokens
        self.dim = dim

    def forward_features(self, x):
        import torch
        g = torch.Generator().manual_seed(0)
        b = x.shape[0]
        return {
            "x_norm_clstoken": torch.randn(b, self.dim, generator=g),
            "x_norm_patchtokens": torch.randn(b, self.n_tokens, self.dim,
                                              generator=g),
        }


def test_embed_runs_on_injected_model_and_builds_norm_stats():
    pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    emb = DinoV2ClsGemEmbedder(device="cpu", model=_FakeDino(n_tokens=4),
                               gem_tokens="all")
    assert emb._mean is None
    rgb = np.full((224, 224, 3), 128, dtype=np.uint8)
    mask = np.zeros((224, 224), dtype=bool)
    mask[64:160, 64:160] = True

    cls, gem = emb.embed(rgb, mask)

    assert emb._mean is not None          # _ensure_norm_stats ran
    assert cls.shape == (8,) and gem.shape == (8,)
    assert abs(float((cls ** 2).sum()) - 1.0) < 1e-6   # cls is L2-normed


def test_fg_patches_runs_on_injected_model_and_builds_norm_stats():
    pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    grid = 4
    ex = DinoV2ForegroundPatchExtractor(device="cpu", grid=grid,
                                        model=_FakeDino(n_tokens=grid * grid))
    assert ex._mean is None
    rgb = np.full((224, 224, 3), 128, dtype=np.uint8)
    mask = np.zeros((224, 224), dtype=bool)
    mask[:, :112] = True                  # half the image: several grid cells

    tok = ex.fg_patches(rgb, mask)

    assert ex._mean is not None           # _ensure_norm_stats ran
    assert tok.ndim == 2 and tok.shape[1] == 8
    assert tok.shape[0] > 0               # some foreground cells survived
