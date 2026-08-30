"""The fallback contract: a chain routes around UNAVAILABLE segmentors, records
which one actually ran, and never hides a real failure.

Pure numpy — no GPU, no SAM2, no DINOv2.
"""

import numpy as np
import pytest

from popoe.interfaces import Detection, ObjectModel, Scene, Segmentor
from popoe.segmentor import DepthSegmentor, SegmentorUnavailable


def _scene(depth=None):
    return Scene(rgb=np.zeros((32, 32, 3), np.uint8),
                 depth=np.zeros((32, 32), np.float32) if depth is None else depth,
                 K=np.eye(3), scene_id=1, im_id=2)


def _obj():
    return ObjectModel(obj_id=7, mesh_path="/nonexistent.ply", diameter=0.1)


class _Unavailable:
    source = "needs-a-checkpoint"

    def segment(self, scene, obj):
        raise SegmentorUnavailable("checkpoint not found")


class _Broken:
    source = "broken"

    def segment(self, scene, obj):
        raise RuntimeError("CUDA out of memory")


class _Works:
    source = "works"

    def __init__(self, n=2):
        self.n = n

    def segment(self, scene, obj):
        return [Detection(mask=np.ones((4, 4), bool), score=0.9)
                for _ in range(self.n)]


def test_depth_segmentor_runs_with_no_deps_and_scores_by_area():
    pytest.importorskip("cv2")   # the only heavy dep below the chain layer
    depth = np.zeros((32, 32), np.float32)
    depth[4:20, 4:20] = 1.0      # big blob: 256 px
    depth[24:30, 24:30] = 1.05   # small blob: 36 px
    dets = DepthSegmentor(min_pixels=10, kernel=3).segment(_scene(depth), _obj())

    assert len(dets) == 2
    assert dets[0].score > dets[1].score            # biggest blob first
    assert dets[0].mask.sum() > dets[1].mask.sum()
    assert {d.source for d in dets} == {"depth-cc"}
    # score IS the area fraction — not a confidence, and not comparable to a
    # DINO cosine similarity. Documented so nobody sorts the two together again.
    h, w = depth.shape
    assert dets[0].score == pytest.approx(dets[0].mask.sum() / (h * w))


def test_segmentors_satisfy_the_stage_protocol():
    assert isinstance(DepthSegmentor(), Segmentor)


def test_an_oom_during_model_load_is_not_treated_as_unavailability(monkeypatch):
    """The load guards are broad on purpose (`except Exception`), because the
    ways a checkpoint can be unreachable are many. That breadth also catches
    out-of-memory — and an OOM reported as SegmentorUnavailable is the worst
    case the contract exists to prevent: the chain routes around it, a weaker
    method answers, and the run looks fine.

    Only torch >= 1.13 raises the dedicated `torch.cuda.OutOfMemoryError`; a
    bare RuntimeError carrying the message is the shape that used to slip
    through.
    """
    torch = pytest.importorskip("torch")

    def _oom(*a, **kw):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(torch.hub, "load", _oom)

    from popoe.segmentor_cnos_lab import DinoV2ForegroundPatchExtractor

    for loader in (lambda: DinoV2ForegroundPatchExtractor(device="cpu").model,):
        with pytest.raises(RuntimeError) as excinfo:
            loader()
        assert not isinstance(excinfo.value, SegmentorUnavailable)
        assert "out of memory" in str(excinfo.value).lower()


def test_a_missing_hub_cache_is_still_unavailability(monkeypatch):
    """The other half of the same guard: a genuinely absent backend must stay
    routable, or the chain has nothing left to do."""
    torch = pytest.importorskip("torch")

    def _no_network(*a, **kw):
        raise OSError("Couldn't reach the hub and no cache entry exists")

    monkeypatch.setattr(torch.hub, "load", _no_network)

    from popoe.segmentor_cnos_lab import DinoV2ForegroundPatchExtractor

    with pytest.raises(SegmentorUnavailable):
        DinoV2ForegroundPatchExtractor(device="cpu").model


def test_cuda_availability_errors_stay_routable():
    """Narrowness matters as much as breadth. Most `CUDA error: ...` messages
    are unavailability, not a runtime fault — including the arch mismatch this
    project actually hits when a pod comes up on a card the kernels were not
    compiled for. Treating them as fatal would break the chain on exactly the
    boxes it exists for."""
    from popoe.interfaces import is_runtime_failure

    for msg in ("CUDA error: invalid device ordinal",
                "CUDA error: no CUDA-capable device is detected",
                "CUDA error: no kernel image is available for execution on the device",
                "CUDA driver version is insufficient for CUDA runtime version"):
        assert not is_runtime_failure(RuntimeError(msg)), msg

    # ... while allocation failures in any spelling stay fatal
    for msg in ("CUDA out of memory. Tried to allocate 2.00 GiB",
                "CUDA_ERROR_OUT_OF_MEMORY",
                "CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate"):
        assert is_runtime_failure(RuntimeError(msg)), msg


def test_hard_device_faults_are_not_mistaken_for_unavailability():
    """The allowlist-vs-denylist distinction, on the fault side.

    The routable availability messages are a small stable set; the sticky fault
    side is open-ended. CUDA, cuDNN and cuBLAS runtime-shaped errors should
    therefore surface unless they match a known availability reason.
    """
    from popoe.interfaces import is_runtime_failure

    for msg in ("CUDA error: unspecified launch failure",
                "CUDA error: misaligned address",
                "CUDA error: an illegal instruction was encountered",
                "CUDA error: the launch timed out and was terminated",
                "CUDA error: uncorrectable ECC error encountered",
                "CUDA error: hardware stack error",
                "CUDA error: invalid program counter",
                "CUDA error: unknown error",
                "CUDA error: context is destroyed",
                "cuDNN error: CUDNN_STATUS_INTERNAL_ERROR",
                "cuDNN error: CUDNN_STATUS_EXECUTION_FAILED",
                "cuBLAS error: CUBLAS_STATUS_EXECUTION_FAILED"):
        assert is_runtime_failure(RuntimeError(msg)), msg


def test_an_arch_mismatch_still_routes_to_the_next_segmentor(monkeypatch):
    """The application-level half of the above: a chain must still fall back."""
    torch = pytest.importorskip("torch")

    def _wrong_arch(*a, **kw):
        raise RuntimeError(
            "CUDA error: no kernel image is available for execution on the device")

    monkeypatch.setattr(torch.hub, "load", _wrong_arch)

    from popoe.segmentor_cnos_lab import DinoV2ForegroundPatchExtractor

    class _DinoLoadGuardSegmentor:
        source = "dinov2-load"

        def __init__(self):
            self.backbone = DinoV2ForegroundPatchExtractor(device="cpu")

        def segment(self, scene, obj):
            self.backbone.model
            return []

    with pytest.raises(SegmentorUnavailable):
        _DinoLoadGuardSegmentor().segment(_scene(), _obj())


def test_an_oom_probe_does_not_latch_the_renderer_into_cpu_mode(monkeypatch):
    """Same contract, the renderer stage.

    `_init_nvdiffrast` marks the probe as tried BEFORE attempting it, so an OOM
    that correctly re-raises must also un-latch that flag. Otherwise the caller
    frees VRAM, retries, and gets "already tried, no context" -> the trimesh CPU
    ray-caster, which renders different views and therefore different query
    features. That is the silent degrade, one retry later.

    `gedi` is stubbed because feature_extractor imports it at module scope from
    POPOE_GEDI_PATH; nothing in this test touches it, and stubbing keeps the
    check running on a dep-light box instead of skipping where it matters.
    """
    import sys
    import types

    pytest.importorskip("torch")
    if "gedi" not in sys.modules:
        stub = types.ModuleType("gedi")
        stub.GeDi = object
        monkeypatch.setitem(sys.modules, "gedi", stub)
    fx = pytest.importorskip("popoe.freeze.feature_extractor")

    calls = {"n": 0}

    def _ctx(device=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return "gpu-context"

    fake = types.ModuleType("nvdiffrast.torch")
    fake.RasterizeCudaContext = _ctx
    pkg = types.ModuleType("nvdiffrast")
    pkg.torch = fake
    monkeypatch.setitem(sys.modules, "nvdiffrast", pkg)
    monkeypatch.setitem(sys.modules, "nvdiffrast.torch", fake)

    ex = fx.QueryFeatureExtractor.__new__(fx.QueryFeatureExtractor)
    ex.device = "cuda"
    ex._render_backend_pref = "auto"
    ex._nvd_init_tried = False
    ex._nvd_ctx = None

    with pytest.raises(RuntimeError, match="out of memory"):
        ex._init_nvdiffrast()

    assert ex._init_nvdiffrast() is True, "retry must re-attempt, not answer from state"
    assert ex._nvd_ctx == "gpu-context"
    assert calls["n"] == 2
