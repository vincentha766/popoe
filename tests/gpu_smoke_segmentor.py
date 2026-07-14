"""GPU smoke test for the segmentor/renderer split. Not part of the CPU suite.

Runs the paths that cannot run without a GPU + nvdiffrast + DINOv2 + SAM2, and
asserts the property the refactor is FOR: when a backend is genuinely missing,
the caller's chain routes around it and RECORDS what actually ran.

    python tests/gpu_smoke_segmentor.py

Uses a synthetic mesh + a rendered scene, so it needs no BOP data.
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from popoe.interfaces import ObjectModel, Scene                       # noqa: E402
from popoe.renderer import (                                          # noqa: E402
    NvdiffrastRenderer, RendererUnavailable, TrimeshRenderer, get_renderer,
    load_mesh_for_rendering,
)
from popoe.segmentor import (                                         # noqa: E402
    DepthSegmentor, FirstAvailableSegmentor, SAMSegmentor,
    SegmentorUnavailable,
)
from popoe.segmentor_cnos import (                                    # noqa: E402
    CNOSSegmentor, CNOSTemplateBank, DepthBoxMasker, DinoV2Backbone,
    DinoWindowSegmentor, SAM2BoxMasker,
)

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}", flush=True)
    except Exception as e:
        FAIL.append((name, f"{type(e).__name__}: {e}"))
        print(f"  FAIL  {name}\n          {type(e).__name__}: {e}", flush=True)


def make_mesh(path):
    """A distinctive, non-symmetric mesh in BOP units (mm)."""
    import trimesh
    m = trimesh.creation.annulus(r_min=20, r_max=45, height=30)
    m += trimesh.creation.box(extents=(90, 18, 18))
    m.export(path)
    return path


def render_scene(renderer, mesh_path, cam_scale=2.4):
    """Render the object once to synthesise an RGB-D 'scene' containing it."""
    V, F, N, _, _ = load_mesh_for_rendering(mesh_path)
    radius = float(np.linalg.norm(np.ptp(V, axis=0))) * cam_scale
    cam = np.array([radius * 0.6, radius * 0.5, radius * 0.6], np.float32)
    rgb, depth = renderer.render(V, F, cam, fov_deg=60.0, normals=N)
    return Scene(rgb=rgb, depth=depth.astype(np.float32), K=np.eye(3),
                 scene_id=1, im_id=1)


def main():
    import torch
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}\n",
          flush=True)
    assert torch.cuda.is_available(), "no GPU — this smoke test needs one"

    tmp = tempfile.mkdtemp()
    mesh_path = make_mesh(os.path.join(tmp, "obj_000001.ply"))
    obj = ObjectModel(obj_id=1, mesh_path=mesh_path, diameter=0.1)

    # ── renderer: explicit backend selection ────────────────────────────
    print("[renderer]", flush=True)
    nvd = get_renderer(224, 224, backend="nvdiffrast")
    check("get_renderer('nvdiffrast') -> NvdiffrastRenderer",
          lambda: (isinstance(nvd, NvdiffrastRenderer) and nvd.source == "nvdiffrast")
          or (_ for _ in ()).throw(AssertionError(f"got {type(nvd).__name__}")))
    check("get_renderer('trimesh') -> TrimeshRenderer, source tagged",
          lambda: get_renderer(224, 224, backend="trimesh").source == "trimesh"
          or (_ for _ in ()).throw(AssertionError("bad source")))
    check("get_renderer('bogus') rejects unknown backend",
          lambda: _expect(ValueError, lambda: get_renderer(backend="bogus")))

    scene = render_scene(nvd, mesh_path)
    check("nvdiffrast render produced a non-empty scene",
          lambda: (scene.depth > 0).sum() > 500
          or (_ for _ in ()).throw(AssertionError(f"{(scene.depth>0).sum()} hit px")))

    # The depth map must be METRIC camera-space distance. The pre-fix code
    # returned 1/(triangle_id) — values unrelated to geometry that only worked
    # as a >0 hit test. The object (diameter 0.1) sits at the origin, so hit
    # depths must cluster around the camera's distance to the origin.
    def depth_is_metric():
        V, F, N, _, _ = load_mesh_for_rendering(mesh_path)
        radius = float(np.linalg.norm(np.ptp(V, axis=0))) * 2.4
        cam = np.array([radius * 0.6, radius * 0.5, radius * 0.6], np.float32)
        _, d = nvd.render(V, F, cam, fov_deg=60.0, normals=N)
        hit = d > 0
        assert hit.sum() > 300, f"only {hit.sum()} hit px"
        med = float(np.median(d[hit]))
        cam_dist = float(np.linalg.norm(cam))
        print(f"          median depth {med:.3f} m vs |cam| {cam_dist:.3f} m",
              flush=True)
        assert abs(med - cam_dist) < 0.35 * cam_dist, (
            f"median hit depth {med:.4f} is not near the camera distance "
            f"{cam_dist:.4f} — depth is not metric (1/triangle_id regression?)")
    check("NvdiffrastRenderer depth is metric camera-space distance",
          depth_is_metric)

    # The two renderers are NOT interchangeable — the claim the cache key now encodes.
    def renders_differ():
        cpu_scene = render_scene(TrimeshRenderer(224, 224), mesh_path)
        gpu_hit = (scene.depth > 0).sum()
        cpu_hit = (cpu_scene.depth > 0).sum()
        d = abs(int(gpu_hit) - int(cpu_hit))
        print(f"          nvdiffrast {gpu_hit} hit px vs trimesh {cpu_hit} "
              f"(delta {d})", flush=True)
        assert not np.array_equal(scene.rgb, cpu_scene.rgb), \
            "renders are pixel-identical — the cache-key split would be pointless"
    check("nvdiffrast and trimesh renders DIFFER (justifies the cache key)",
          renders_differ)

    # ── DINOv2 backbone ─────────────────────────────────────────────────
    print("\n[dino]", flush=True)
    dino = DinoV2Backbone(device="cuda")
    check("DinoV2Backbone.cls_token -> (D,)",
          lambda: dino.cls_token(scene.rgb, side=224).shape == (768,)
          or (_ for _ in ()).throw(AssertionError(dino.cls_token(scene.rgb).shape)))
    check("DinoV2Backbone.patch_tokens -> (n_ph, n_pw, D)",
          lambda: dino.patch_tokens(scene.rgb).shape == (16, 16, 768)
          or (_ for _ in ()).throw(AssertionError(dino.patch_tokens(scene.rgb).shape)))

    # ── template bank ───────────────────────────────────────────────────
    print("\n[templates]", flush=True)
    bank = CNOSTemplateBank(nvd, dino, n_templates=12)

    def bank_ok():
        feats = bank.feats_for(obj)
        assert feats.ndim == 2 and feats.shape[0] >= 8, f"only {feats.shape} templates"
        norms = np.linalg.norm(feats, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4), f"not L2-normed: {norms[:3]}"
    check("CNOSTemplateBank renders + embeds, feats are L2-normed", bank_ok)
    check("template bank is cached per obj_id (2nd call is free)",
          lambda: bank.feats_for(obj) is bank.feats_for(obj)
          or (_ for _ in ()).throw(AssertionError("re-rendered")))

    # ── SAM2 availability decides which segmentors can run ──────────────
    ckpt_dir = os.environ.get("POPOE_SAM2_CKPT", "/workspace/sam2_checkpoints")
    have_sam2 = os.path.exists(os.path.join(ckpt_dir, "sam2.1_hiera_small.pt"))
    print(f"\n[sam2] checkpoint present: {have_sam2} ({ckpt_dir})", flush=True)

    # ── DinoWindowSegmentor with the SAM2-free masker ───────────────────
    print("\n[dino-window + depth masker]  (no SAM2 needed)", flush=True)
    win = DinoWindowSegmentor(nvd, masker=DepthBoxMasker(), dino=dino, bank=bank,
                              conf_threshold=-1.0, n_masks=3)

    def window_ok():
        dets = win.segment(scene, obj)
        assert dets, "no detections"
        assert dets[0].source == "dino-window+depth-box", dets[0].source
        assert -1.0 <= dets[0].score <= 1.0, f"score {dets[0].score} not a cosine"
        assert dets[0].mask.sum() > 100, "mask too small"
        assert dets == sorted(dets, key=lambda d: -d.score), "not sorted by score"
        print(f"          {len(dets)} dets, best cos={dets[0].score:.3f}, "
              f"{dets[0].mask.sum()} px, source={dets[0].source}", flush=True)
    check("DinoWindowSegmentor(DepthBoxMasker) segments + stamps source", window_ok)

    # ── the chain: a REAL missing backend must be routed around, and said ──
    # This is the property the whole refactor exists for. Point CNOS at an empty
    # checkpoint dir: it must REFUSE, not quietly produce worse masks.
    print("\n[chain] CNOS with an empty checkpoint dir -> must be UNAVAILABLE", flush=True)
    empty_dir = os.path.join(tmp, "no_checkpoints_here")
    os.makedirs(empty_dir, exist_ok=True)

    def chain_routes_and_records():
        broken = CNOSSegmentor(nvd, dino=dino, bank=bank, sam_ckpt_dir=empty_dir)
        raised = False
        try:
            broken.segment(scene, obj)
        except SegmentorUnavailable as e:
            raised = True
            print(f"          CNOS correctly refused: {str(e).splitlines()[0]}", flush=True)
        assert raised, "CNOS silently degraded instead of raising — the whole bug"

        chain = FirstAvailableSegmentor([broken, win, DepthSegmentor()])
        dets = chain.segment(scene, obj)
        assert dets, "chain produced nothing"
        assert chain.last_used == "dino-window", f"last_used={chain.last_used}"
        assert all(d.source.startswith("dino-window") for d in dets), \
            f"provenance lost: {[d.source for d in dets]}"
        print(f"          chain.last_used={chain.last_used!r}  "
              f"sources={sorted({d.source for d in dets})}", flush=True)
    check("missing SAM2 ckpt -> CNOS raises, chain falls through, provenance recorded",
          chain_routes_and_records)

    # ── the real CNOS path (needs the checkpoint) ───────────────────────
    if have_sam2:
        print("\n[cnos] SAM2 AMG proposals -> DINOv2 rerank", flush=True)
        cnos = CNOSSegmentor(nvd, dino=dino, bank=bank, conf_threshold=-1.0, n_masks=5)

        def cnos_ok():
            dets = cnos.segment(scene, obj)
            assert dets, "no detections"
            assert dets[0].source == "cnos", dets[0].source
            assert -1.0 <= dets[0].score <= 1.0, f"score {dets[0].score} not a cosine"
            assert dets[0].descriptor is not None and dets[0].descriptor.shape == (768,)
            assert dets == sorted(dets, key=lambda d: -d.score), "not sorted"
            print(f"          {len(dets)} dets, best cos={dets[0].score:.3f}, "
                  f"{dets[0].mask.sum()} px", flush=True)
        check("CNOSSegmentor end-to-end (AMG -> rerank)", cnos_ok)

        def sam_ok():
            dets = SAMSegmentor(n_masks=3).segment(scene, obj)
            assert dets and dets[0].source == "sam2-amg"
            assert 0.0 <= dets[0].score <= 1.0, "predicted_iou out of range"
        check("SAMSegmentor end-to-end", sam_ok)

        def sam_box_ok():
            w2 = DinoWindowSegmentor(nvd, masker=SAM2BoxMasker(), dino=dino,
                                     bank=bank, conf_threshold=-1.0, n_masks=3)
            dets = w2.segment(scene, obj)
            assert dets and dets[0].source == "dino-window+sam2-box", dets[0].source
        check("DinoWindowSegmentor(SAM2BoxMasker) — masker is injectable", sam_box_ok)

        def chain_prefers_cnos():
            chain = FirstAvailableSegmentor([cnos, win, DepthSegmentor()])
            dets = chain.segment(scene, obj)
            assert chain.last_used == "cnos", chain.last_used
            assert all(d.source == "cnos" for d in dets)
        check("chain prefers CNOS when SAM2 IS available", chain_prefers_cnos)
    else:
        print("  SKIP  CNOS / SAM paths — no SAM2 checkpoint", flush=True)

    # ── depth segmentor on a real depth map ─────────────────────────────
    print("\n[depth]", flush=True)
    check("DepthSegmentor works on a real depth map, scores by area",
          lambda: _depth_ok(scene, obj))

    # ── feature_extractor render backend is reportable (-> cache key) ───
    print("\n[feature_extractor]", flush=True)

    def backend_reported():
        try:
            from popoe.feature_extractor import QueryFeatureExtractor
        except ImportError as e:
            # feature_extractor does `from gedi import GeDi` at module level;
            # render_backend needs neither gedi nor dino, so a missing
            # eval-only dep is an environment gap here, not a logic failure.
            missing = getattr(e, "name", None) or ""
            if missing == "popoe" or missing.startswith("popoe."):
                raise  # our own module is broken — that IS a failure
            print(f"  SKIP  QueryFeatureExtractor.render_backend — "
                  f"eval-only dep missing: {e}", flush=True)
            return
        # inject dummies: we are testing backend resolution, not the models
        qx = QueryFeatureExtractor(device="cuda", dino=object(), gedi=object(),
                                   render_backend="auto")
        assert qx.render_backend == "nvdiffrast", qx.render_backend
        cpu = QueryFeatureExtractor(device="cuda", dino=object(), gedi=object(),
                                    render_backend="trimesh")
        assert cpu.render_backend == "trimesh", cpu.render_backend
        print(f"          auto -> {qx.render_backend!r}, forced -> {cpu.render_backend!r}"
              "  (both now enter the cache key)", flush=True)
    check("QueryFeatureExtractor.render_backend is reportable", backend_reported)

    print(f"\n{'='*66}\n{len(PASS)} passed, {len(FAIL)} failed", flush=True)
    for name, err in FAIL:
        print(f"  FAILED: {name}\n          {err}", flush=True)
    return 1 if FAIL else 0


def _expect(exc, fn):
    try:
        fn()
    except exc:
        return True
    raise AssertionError(f"expected {exc.__name__}")


def _depth_ok(scene, obj):
    dets = DepthSegmentor(min_pixels=50).segment(scene, obj)
    assert dets, "no depth components"
    assert dets[0].source == "depth-cc"
    h, w = scene.depth.shape
    assert abs(dets[0].score - dets[0].mask.sum() / (h * w)) < 1e-6, \
        "score is not the area fraction it is documented to be"
    print(f"          {len(dets)} components, biggest area frac={dets[0].score:.3f}",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
