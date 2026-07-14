"""Object-agnostic segmentors + the explicit fallback chain.

Every segmentor here satisfies `interfaces.Segmentor`
(`segment(scene, obj) -> list[Detection]`) and does EXACTLY ONE thing. A
segmentor whose dependency is missing raises `SegmentorUnavailable`; it never
quietly substitutes a weaker method.

Why no fallback inside an implementation
----------------------------------------
It used to be inside: `SAMSegmentor.segment` silently dropped to depth
connected-components on load failure, on generate() exception, AND on an empty
result; `CNOSSegmentor` silently dropped to a sliding-window variant. Two
things that costs you:

  * **You cannot tell what actually ran.** A run on a box without the SAM2
    checkpoint produced depth-blob masks while every log line, config and
    cache key still said "CNOS". `Detection.source` + `FirstAvailableSegmentor
    .last_used` now make the answer explicit and machine-readable.
  * **Scores stop being comparable.** `score` means a different thing per
    method (cosine similarity / SAM predicted-IoU / area fraction). The old
    CNOS fallback appended depth-blob AREA FRACTIONS into a list of DINO
    COSINE SIMILARITIES and sorted the two together — a blob covering 40% of
    the frame (0.40) outranked a genuine template match (0.35). Splitting the
    methods keeps one score semantics per list; the chain returns one
    segmentor's output, never a blend.

Fallback is a CALLER's policy, so the caller composes it and can see the
outcome:

    seg = FirstAvailableSegmentor([
        CNOSSegmentor(renderer),      # SAM2 + DINOv2 templates
        DepthSegmentor(),             # no deps; always available
    ])
    dets = seg.segment(scene, obj)
    print(seg.last_used)              # -> "cnos" or "depth-cc"

Only `SegmentorUnavailable` (missing package / missing checkpoint) advances the
chain. A runtime failure — CUDA OOM, a corrupt image — propagates: masking real
bugs as "the fallback handled it" is how the above shipped in the first place.

Installation (SAM2 paths):
    pip install git+https://github.com/facebookresearch/sam2.git
    wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt \
         -P /workspace/sam2_checkpoints/
"""

from __future__ import annotations

import os

import numpy as np

from popoe.interfaces import BackendUnavailable, Detection, ObjectModel, Scene

# Heavy deps (torch/sam2/cv2) are imported inside the methods that need them, per
# the lazy-import policy in popoe/__init__.py — so the chain and the protocol
# stay importable, and unit-testable, on a box with numpy alone.

torch_cudnn_disabled = False  # set on first heavy import; see _disable_cudnn()

# SAM2.1 configs (official sam-2 >= 1.0) and matching checkpoints (092824 release).
# Single source of truth — segmentor_cnos.py imports these rather than re-listing.
SAM2_CONFIGS = {
    'tiny':  ('configs/sam2.1/sam2.1_hiera_t.yaml',  'sam2.1_hiera_tiny.pt'),
    'small': ('configs/sam2.1/sam2.1_hiera_s.yaml',  'sam2.1_hiera_small.pt'),
    'base':  ('configs/sam2.1/sam2.1_hiera_b+.yaml', 'sam2.1_hiera_base_plus.pt'),
    'large': ('configs/sam2.1/sam2.1_hiera_l.yaml',  'sam2.1_hiera_large.pt'),
}
AMG_PARAMS = dict(points_per_side=32, pred_iou_thresh=0.7,
                  stability_score_thresh=0.85, box_nms_thresh=0.7)


def default_ckpt_dir() -> str:
    """Read POPOE_SAM2_CKPT at CALL time, not import time — a module-level
    constant freezes whatever the env happened to be when popoe was first
    imported, which is invisible and untestable."""
    return os.environ.get('POPOE_SAM2_CKPT', '/workspace/sam2_checkpoints')


class SegmentorUnavailable(BackendUnavailable):
    """A segmentor's backend is missing (package, checkpoint, device).

    The ONLY condition that advances a FirstAvailableSegmentor chain. Runtime
    errors are not this: they propagate. See interfaces.BackendUnavailable."""


def _disable_cudnn() -> None:
    """cudnn breaks SAM2's Hiera backbone on some driver combos. Applied on
    first model build rather than at import (a module import must not mutate
    global torch state — the previous version did, for every importer)."""
    global torch_cudnn_disabled
    if not torch_cudnn_disabled:
        import torch
        torch.backends.cudnn.enabled = False
        torch_cudnn_disabled = True


def build_sam2_model(model_size: str = 'small', device: str = 'cuda',
                     ckpt_dir: str | None = None):
    """Build a SAM2.1 backbone, or raise SegmentorUnavailable.

    Missing package / missing checkpoint are *unavailability*, not bugs — they
    are the two conditions a caller's fallback chain is allowed to route around.
    """
    _disable_cudnn()
    try:
        from sam2.build_sam import build_sam2
    except ImportError as e:
        raise SegmentorUnavailable(
            "sam2 not installed: pip install git+https://github.com/"
            "facebookresearch/sam2.git") from e

    ckpt_dir = ckpt_dir or default_ckpt_dir()
    cfg, ckpt_name = SAM2_CONFIGS[model_size]
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        raise SegmentorUnavailable(
            f"SAM2 checkpoint not found: {ckpt_path}\n"
            f"  mkdir -p {ckpt_dir}\n"
            f"  wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
            f"{ckpt_name} -P {ckpt_dir}/")
    return build_sam2(cfg, ckpt_path, device=device, apply_postprocessing=False)


# ── The explicit chain ──────────────────────────────────────────────────

class FirstAvailableSegmentor:
    """Try each segmentor in order; use the first one that is available.

    This is the ONLY place a fallback lives. It records what ran:
    `last_used` (str) and `Detection.source` on every returned detection.

    Args:
        segmentors: ordered [(name, segmentor)] or [segmentor] (name taken
            from `.source` if the segmentor exposes one, else its class name).
        advance_on_empty: if True, an available segmentor returning NO
            detections also advances the chain. Default False — "no object in
            this image" is a legitimate answer, and conflating it with "this
            segmentor is broken" is what the old SAMSegmentor did.
    """

    def __init__(self, segmentors, advance_on_empty: bool = False):
        self.segmentors = [
            s if isinstance(s, tuple)
            else (getattr(s, 'source', None) or type(s).__name__, s)
            for s in segmentors
        ]
        self.advance_on_empty = advance_on_empty
        self.last_used: str | None = None
        self._skipped: dict[str, str] = {}   # name -> why unavailable

    def segment(self, scene: Scene, obj: ObjectModel) -> list[Detection]:
        for name, seg in self.segmentors:
            try:
                dets = seg.segment(scene, obj)
            except SegmentorUnavailable as e:
                if name not in self._skipped:     # announce once, not per image
                    self._skipped[name] = str(e)
                    print(f"[segmentor] {name} unavailable -> {e}\n"
                          f"[segmentor] falling back to the next in the chain.")
                continue
            if not dets and self.advance_on_empty:
                continue
            self.last_used = name
            for d in dets:
                d.source = d.source or name
            return dets
        self.last_used = None
        raise SegmentorUnavailable(
            "no segmentor in the chain was available: "
            + "; ".join(f"{n}: {w}" for n, w in self._skipped.items()))


# ── Segmentors ──────────────────────────────────────────────────────────

class SAMSegmentor:
    """SAM2.1 automatic mask generator. Class-agnostic: returns whatever the
    scene contains, ranked by SAM's predicted IoU, with NO regard for `obj`.

    `score` is SAM's predicted_iou (mask quality), NOT a match confidence —
    use CNOSSegmentor if you need candidates ranked by resemblance to the CAD
    model."""

    source = 'sam2-amg'

    def __init__(self, device: str = 'cuda', model_size: str = 'small',
                 n_masks: int = 5, conf_threshold: float = 0.4,
                 min_mask_region_area: int = 200,
                 sam_ckpt_dir: str | None = None):
        self.device = device
        self.model_size = model_size
        self.n_masks = n_masks
        self.conf_threshold = conf_threshold
        self.min_mask_region_area = min_mask_region_area
        self.sam_ckpt_dir = sam_ckpt_dir
        self._generator = None

    def _load(self):
        if self._generator is None:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            model = build_sam2_model(self.model_size, self.device, self.sam_ckpt_dir)
            self._generator = SAM2AutomaticMaskGenerator(
                model, min_mask_region_area=self.min_mask_region_area,
                **AMG_PARAMS)
            print(f"SAM2.1-{self.model_size} AMG loaded.")
        return self._generator

    def segment(self, scene: Scene, obj: ObjectModel) -> list[Detection]:
        amg = self._load()                      # raises SegmentorUnavailable
        proposals = amg.generate(scene.rgb)     # runtime errors propagate
        dets = []
        for m in sorted(proposals, key=lambda x: -x['predicted_iou']):
            if float(m['predicted_iou']) < self.conf_threshold:
                continue
            dets.append(Detection(mask=m['segmentation'].astype(bool),
                                  score=float(m['predicted_iou']),
                                  source=self.source))
            if len(dets) >= self.n_masks:
                break
        return dets


class DepthSegmentor:
    """Depth connected-components. No learned model, no checkpoint, no GPU —
    the always-available last resort of a fallback chain.

    `score` is the mask's AREA FRACTION, not a confidence: this segmentor has
    no notion of the query object, so it cannot rank by resemblance. Biggest
    blob first. Do not compare these scores against any other segmentor's."""

    source = 'depth-cc'

    def __init__(self, n_masks: int = 5, depth_band: tuple = (0.7, 1.3),
                 min_pixels: int = 100, kernel: int = 5):
        self.n_masks = n_masks
        self.depth_band = depth_band
        self.min_pixels = min_pixels
        self.kernel = kernel

    def segment(self, scene: Scene, obj: ObjectModel) -> list[Detection]:
        import cv2
        depth = scene.depth
        if depth is None:
            return []
        valid = depth > 0
        if not valid.any():
            return []
        med = float(np.median(depth[valid]))
        lo, hi = self.depth_band
        mask = valid & (depth > med * lo) & (depth < med * hi)

        k = np.ones((self.kernel, self.kernel), np.uint8)
        m8 = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k)
        m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, k)

        n_labels, labels, _, _ = cv2.connectedComponentsWithStats(m8)
        h, w = depth.shape
        dets = []
        for lbl in range(1, n_labels):
            comp = labels == lbl
            if comp.sum() >= self.min_pixels:
                dets.append(Detection(mask=comp, score=float(comp.sum()) / (h * w),
                                      source=self.source))
        dets.sort(key=lambda d: -d.score)
        return dets[: self.n_masks]


def get_dense_target_pcd(depth, mask, intrinsics):
    """Build dense point cloud from depth map within mask."""
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
    ys, xs = np.where(mask & (depth > 0))
    d = depth[ys, xs]
    X = (xs - cx) * d / fx
    Y = (ys - cy) * d / fy
    return np.stack([X, Y, d], axis=1).astype(np.float32)
