"""popoe.assembly — load each heavy model once, share it across components.

Motivation (measured, DEMO_SINGLE_GPU.md): DINOv2 ViT-g/14 holds ~4.3 GB of
VRAM and three components load their own copy — MUSE's crop embedder, CNOS-lab's
patch extractor, and the pose-side query/target encoders — with SAM2 loaded
twice more (MUSE's box refiner, the AMG proposers). Run as independent
processes, the live segmentation ensemble plus pose does not fit a 24 GB card;
with one copy of each model it fits with room to spare. The redundancy is
structural, so the fix is structural: one pool, many components.

``ModelPool`` owns one lazily-loaded instance of each shared model. The
``*_from_pool`` builders wire existing components around it through their
``model=`` / ``sam_model=`` / ``dino=`` / ``gedi=`` injection parameters.

Two deliberate departures from popoe's usual lazy style, both scoped to this
module: the pool getters load at *wiring* time (a long-lived process wants its
weights resident at startup, and a broken backend should surface then, not on
frame one); and sharing assumes the *sequential* frame loop popoe already has —
the shared modules are stateless in forward, but popoe adds no locking, so
concurrent segmentors need their own synchronisation.

Boundary note: this module is library-side composition only. The service shell
(HTTP, process supervision, cameras) lives outside popoe — see AGENTS.md.
"""
from __future__ import annotations

from typing import Optional

from popoe.interfaces import is_runtime_failure
from popoe.segmentor import SegmentorUnavailable, build_sam2_model


class ModelPool:
    """One lazily-loaded copy of each shared heavy model.

    Getters cache: every caller receives the SAME instance. The pool pins the
    identity knobs (device, model names) so components built from it cannot
    disagree with the weights they share — component ``config()`` output keeps
    describing models by name, which stays truthful because the pool is the
    single place that resolves those names to weights.

    The ``_load_*`` methods are seams: tests and exotic deployments override
    them per-instance to control what a getter hands out.
    """

    def __init__(self, device: str = "cuda",
                 dinov2_model: str = "dinov2_vitg14_reg",
                 sam2_size: str = "large",
                 sam_ckpt_dir: Optional[str] = None):
        self.device = device
        self.dinov2_model = dinov2_model
        self.sam2_size = sam2_size
        self.sam_ckpt_dir = sam_ckpt_dir
        self._dinov2 = None
        self._sam2 = None
        self._gedi = None

    def config(self) -> dict:
        # Same rationale as SAM2BoxMaskRefiner.config(): resolve the checkpoint
        # dir NOW, so two pools under different POPOE_SAM2_CKPT values never
        # share an identity while loading different weights.
        from popoe.segmentor import default_ckpt_dir
        return {"device": self.device, "dinov2_model": self.dinov2_model,
                "sam2_size": self.sam2_size,
                "sam_ckpt_dir": self.sam_ckpt_dir or default_ckpt_dir()}

    # ── loader seams ────────────────────────────────────────────────────

    def _load_dinov2(self):
        try:
            import torch
        except ImportError as e:
            raise SegmentorUnavailable(f"torch not installed: {e}") from e
        try:
            return torch.hub.load(
                "facebookresearch/dinov2", self.dinov2_model, pretrained=True
            ).to(self.device).eval()
        except Exception as e:
            if is_runtime_failure(e):
                raise
            raise SegmentorUnavailable(
                f"DINOv2 ({self.dinov2_model}) load failed: {e}") from e

    def _load_sam2(self):
        return build_sam2_model(self.sam2_size, self.device, self.sam_ckpt_dir)

    def _load_gedi(self):
        # Module-level heavy: importing popoe.freeze.feature_extractor needs
        # the GeDi package reachable via POPOE_GEDI_PATH, so defer it to here.
        from popoe.freeze.feature_extractor import load_gedi
        return load_gedi(self.device)

    # ── cached getters ──────────────────────────────────────────────────

    def dinov2(self):
        if self._dinov2 is None:
            self._dinov2 = self._load_dinov2()
        return self._dinov2

    def sam2(self):
        if self._sam2 is None:
            self._sam2 = self._load_sam2()
        return self._sam2

    def gedi(self):
        if self._gedi is None:
            self._gedi = self._load_gedi()
        return self._gedi


# ── wiring helpers ──────────────────────────────────────────────────────

def muse_segmentor_from_pool(pool: ModelPool, classes, *,
                             mask_rgb: bool = True,
                             gem_tokens: str = "all",
                             **kwargs):
    """``build_muse_segmentor`` on the pool's DINOv2 + SAM2.

    Grounding DINO stays per-segmentor — nothing else in popoe uses it, so
    there is no copy to save. Remaining ``build_muse_segmentor`` knobs (prompt,
    thresholds, size gate…) pass through ``kwargs``.
    """
    from popoe.segmentor_muse import (DinoV2ClsGemEmbedder, SAM2BoxMaskRefiner,
                                      build_muse_segmentor)
    embedder = DinoV2ClsGemEmbedder(
        device=pool.device, model_name=pool.dinov2_model,
        mask_rgb=mask_rgb, gem_tokens=gem_tokens, model=pool.dinov2())
    refiner = SAM2BoxMaskRefiner(pool.device, pool.sam2_size,
                                 pool.sam_ckpt_dir, sam_model=pool.sam2())
    return build_muse_segmentor(classes, device=pool.device,
                                embedder=embedder, refiner=refiner, **kwargs)


def cnos_lab_segmentor_from_pool(pool: ModelPool, template_dir, *,
                                min_mask_region_area: int = 2000,
                                **kwargs):
    """``CNOSLabSegmentor`` on the pool's DINOv2 + SAM2.

    ``template_dir`` is anything ``TemplateDirPatchBank`` accepts (str,
    ``{obj_id: dir}``, or callable). Remaining ``CNOSLabSegmentor`` knobs
    (size_gate, n_masks, conf_threshold) pass through ``kwargs``. Note the AMG
    runs the pool's ``sam2_size`` checkpoint, not this class's historical
    'small' default.
    """
    from popoe.segmentor_cnos_lab import (CNOSLabSegmentor,
                                          DinoV2ForegroundPatchExtractor,
                                          SAM2AMGMaskProposer,
                                          TemplateDirPatchBank)
    extractor = DinoV2ForegroundPatchExtractor(
        device=pool.device, model_name=pool.dinov2_model, model=pool.dinov2())
    proposer = SAM2AMGMaskProposer(
        device=pool.device, model_size=pool.sam2_size,
        min_mask_region_area=min_mask_region_area,
        sam_ckpt_dir=pool.sam_ckpt_dir, sam_model=pool.sam2())
    return CNOSLabSegmentor(proposer=proposer,
                           template_bank=TemplateDirPatchBank(template_dir,
                                                              extractor),
                           **kwargs)


def pose_encoders_from_pool(pool: ModelPool):
    """``(QueryFeatureExtractor, TargetFeatureExtractor)`` sharing one DINOv2
    and one GeDi.

    The FreeZe recipe hard-codes ``dinov2_vitg14_reg`` in its own loader, so a
    pool with any other ``dinov2_model`` is refused rather than silently
    changing the reproduction's visual backbone.
    """
    if pool.dinov2_model != "dinov2_vitg14_reg":
        raise ValueError(
            "pose encoders expect dinov2_vitg14_reg (the FreeZe recipe), "
            f"pool has {pool.dinov2_model!r}")
    from popoe.freeze.feature_extractor import (QueryFeatureExtractor,
                                                TargetFeatureExtractor)
    dino, gedi = pool.dinov2(), pool.gedi()
    return (QueryFeatureExtractor(device=pool.device, dino=dino, gedi=gedi),
            TargetFeatureExtractor(device=pool.device, dino=dino, gedi=gedi))


__all__ = [
    "ModelPool",
    "muse_segmentor_from_pool",
    "cnos_lab_segmentor_from_pool",
    "pose_encoders_from_pool",
]
