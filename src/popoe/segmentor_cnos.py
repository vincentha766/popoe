"""CNOS-style segmentors: DINOv2 template matching against CAD renders.

Reproduces the core idea of CNOS: A Strong Baseline for CAD-novel Object
Segmentation (ICCV-W 2023) — render the CAD model from many viewpoints, embed
each view with DINOv2, then rank scene regions by cosine similarity to those
templates.

TWO segmentors live here. They are siblings, NOT a primary and its fallback:
they differ in where the candidate regions come from, and that changes the
result quality enough that you must know which one ran.

  CNOSSegmentor        SAM2 AMG proposes every instance in the scene, DINOv2
                       re-ranks each against the templates. This is the paper's
                       pipeline. Needs sam2 + a checkpoint.
  DinoWindowSegmentor  No SAM2 needed for proposals: slide multi-scale windows
                       over the DINOv2 patch-feature grid and keep the
                       best-matching boxes, then turn each box into a mask with
                       an injected `masker` (SAM2 box prompt, or a depth band).
                       Coarser — window boxes are axis-aligned and quantised to
                       14-px patches — but it runs with no SAM2 at all when
                       paired with DepthBoxMasker.

Which one ran used to be invisible: CNOSSegmentor.segment() caught a SAM2 load
failure and silently called the window variant, which itself silently swapped
its masker, and then topped the list up with depth blobs whose "score" was a
mask AREA FRACTION mixed in among DINO COSINE SIMILARITIES. Compose them
explicitly instead, and the answer is in `Detection.source`:

    seg = FirstAvailableSegmentor([
        CNOSSegmentor(renderer),
        DinoWindowSegmentor(renderer, masker=DepthBoxMasker()),
    ])

TEMPLATE CAVEAT (unchanged, and the reason self-rendered templates are not
always the right source): templates come from the flat Lambertian, *untextured*
renderer (renderer.NvdiffrastRenderer: single light, constant base_color, no
albedo/UV). That is FAITHFUL for LMO — LineMod CAD has no UV texture, so the
paper renders it Lambertian too, and LMO self-segmentation via this path
legitimately matches the paper's CNOS line. It is INSUFFICIENT for textured
objects (YCB-V, HOPE, ...): DINOv2 then matches on silhouette only, losing
logo/label/colour cues. For textured datasets read the official CNOS-FastSAM
BOP detection JSON via segmentor_detections.BOPDetectionsSegmentor instead.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from popoe.interfaces import Detection, ObjectModel, Scene
from popoe.segmentor import (
    AMG_PARAMS, SegmentorUnavailable, _disable_cudnn, build_sam2_model,
)

# torch / cv2 / PIL / torchvision are imported inside the methods that need
# them: composing a FirstAvailableSegmentor chain REQUIRES importing this
# module, so it must import on a box without the heavy deps — otherwise the
# chain can never route around their absence.

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_PATCH = 14   # DINOv2 ViT-*/14


# ── DINOv2 backbone (shared) ────────────────────────────────────────────

class DinoV2Backbone:
    """Lazily-loaded DINOv2 ViT-B/14. Pass ONE instance to both segmentors and
    the template bank so the weights are loaded once.

    Load failure raises SegmentorUnavailable (a chain may route around a box
    with no hub cache and no network). Inference errors propagate — those are
    bugs, not unavailability."""

    def __init__(self, device: str = 'cuda', name: str = 'dinov2_vitb14_reg'):
        self.device = device
        self.name = name
        self._model = None
        self._tf = None

    @property
    def model(self):
        if self._model is None:
            try:
                import torch
            except ImportError as e:
                raise SegmentorUnavailable(f"torch not installed: {e}") from e
            _disable_cudnn()      # cuDNN init fails on some hosts (see feature_extractor)
            try:
                self._model = torch.hub.load(
                    'facebookresearch/dinov2', self.name, pretrained=True
                ).to(self.device).eval()
            except Exception as e:                    # no network / no hub cache
                raise SegmentorUnavailable(f"DINOv2 ({self.name}) load failed: {e}") from e
        return self._model

    def _to_tensor(self, img_rgb: np.ndarray, size: tuple[int, int]):
        """size = (H, W), each a multiple of 14."""
        from torchvision import transforms
        import PIL.Image
        if self._tf is None:
            self._tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ])
        h, w = size
        pil = PIL.Image.fromarray(img_rgb).resize((w, h))
        return self._tf(pil).unsqueeze(0).to(self.device)

    def cls_token(self, img_rgb: np.ndarray, side: int = 224) -> np.ndarray:
        """Global descriptor of a (square-ish) image. Returns (D,)."""
        import torch
        model = self.model
        with torch.no_grad():
            img_t = self._to_tensor(img_rgb, (side, side))
            out = model.get_intermediate_layers(
                img_t, n=[model.n_blocks - 1], return_class_token=True)
            return out[0][1].squeeze(0).cpu().numpy()

    def patch_tokens(self, img_rgb: np.ndarray) -> np.ndarray:
        """Dense patch grid of a full image. Returns (n_ph, n_pw, D)."""
        import torch
        model = self.model
        with torch.no_grad():
            h, w = img_rgb.shape[:2]
            h_r, w_r = (h // _PATCH) * _PATCH, (w // _PATCH) * _PATCH
            img_t = self._to_tensor(img_rgb, (h_r, w_r))
            out = model.get_intermediate_layers(
                img_t, n=[model.n_blocks - 1], return_class_token=False)[0]
            return out[0].reshape(h_r // _PATCH, w_r // _PATCH, -1).cpu().numpy()


def _l2norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


def _masked_square_crop(rgb: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    """Square crop around the mask with non-mask pixels blacked out (the CNOS
    trick — the descriptor must describe the object, not its background)."""
    ys, xs = np.where(mask)
    if len(ys) < 10:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = max(y1 - y0, x1 - x0)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    h_img, w_img = rgb.shape[:2]
    y0p = max(0, cy - side // 2); y1p = min(h_img, y0p + side)
    x0p = max(0, cx - side // 2); x1p = min(w_img, x0p + side)
    crop = rgb[y0p:y1p, x0p:x1p].copy()
    crop[~mask[y0p:y1p, x0p:x1p]] = 0
    return crop


# ── Template bank ───────────────────────────────────────────────────────

class CNOSTemplateBank:
    """Renders a CAD model from N icosphere-ish viewpoints and embeds each view
    with DINOv2. Feats are L2-normed and cached per (obj_id, mesh_path), so the
    two segmentors below can share one bank.

    KNOWN DEVIATION (pre-existing, preserved deliberately): templates are
    embedded at the render resolution — (renderer.H // 14) * 14, i.e. 476 by
    default — while scene crops are embedded at 224 (CNOS uses 224 for both).
    The CLS tokens are therefore compared across a scale gap. Kept as-is because
    changing it shifts every similarity score, and this segmentor has no
    evaluated baseline to check that against; fix it deliberately, with numbers,
    not as a drive-by."""

    def __init__(self, renderer, dino: DinoV2Backbone, n_templates: int = 42,
                 fov_deg: float = 60.0):
        self.renderer = renderer
        self.dino = dino
        self.n_templates = n_templates
        self.fov_deg = fov_deg
        # Keyed by (obj_id, mesh_path), not obj_id alone: BOP object ids are
        # only unique within one dataset, and one bank instance may serve two.
        self._banks: dict[tuple, np.ndarray] = {}

    def feats_for(self, obj: ObjectModel) -> np.ndarray:
        bank_key = (obj.obj_id, obj.mesh_path)
        if bank_key in self._banks:
            return self._banks[bank_key]

        from popoe.renderer import fibonacci_viewpoints, load_mesh_for_rendering

        V, F, N, _scale, _center = load_mesh_for_rendering(obj.mesh_path)
        radius = float(np.linalg.norm(np.ptp(V, axis=0))) * 1.5
        side = (self.renderer.H // _PATCH) * _PATCH

        print(f"obj{obj.obj_id}: rendering {self.n_templates} templates...")
        feats = []
        for cam_pos in fibonacci_viewpoints(self.n_templates, radius):
            rgb, depth = self.renderer.render(V, F, cam_pos, fov_deg=self.fov_deg,
                                              normals=N)
            if (depth > 0).sum() < 100:            # blank render, skip
                continue
            feats.append(self.dino.cls_token(rgb, side=side))
        if not feats:
            raise RuntimeError(
                f"obj{obj.obj_id}: every template render was blank — check the "
                f"renderer and {obj.mesh_path}")

        bank = _l2norm(np.stack(feats, axis=0), axis=1)
        print(f"obj{obj.obj_id}: templates ready ({len(feats)}/{self.n_templates} valid)")
        self._banks[bank_key] = bank
        return bank


# ── Box -> mask strategies (for DinoWindowSegmentor) ────────────────────

class SAM2BoxMasker:
    """Turn a box into a mask with a SAM2 box prompt. Sharp boundaries."""

    source = 'sam2-box'

    def __init__(self, device: str = 'cuda', model_size: str = 'small',
                 sam_ckpt_dir: Optional[str] = None):
        self.device = device
        self.model_size = model_size
        self.sam_ckpt_dir = sam_ckpt_dir
        self._predictor = None

    def _load(self):
        if self._predictor is None:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            self._predictor = SAM2ImagePredictor(
                build_sam2_model(self.model_size, self.device, self.sam_ckpt_dir))
            print(f"SAM2.1-{self.model_size} box predictor loaded.")
        return self._predictor

    def masks_for(self, scene: Scene, boxes) -> list[Optional[np.ndarray]]:
        p = self._load()                      # raises SegmentorUnavailable
        # Embed once per call, covering every box. NOT cached across calls keyed
        # on Scene ids: those default to -1, so two different images can collide
        # and silently reuse a stale embedding.
        p.set_image(scene.rgb)
        out = []
        for (y0, x0, y1, x1) in boxes:
            masks, scores, _ = p.predict(box=np.array([x0, y0, x1, y1]),
                                         multimask_output=True)
            out.append(masks[scores.argmax()].astype(bool))
        return out


class DepthBoxMasker:
    """Turn a box into a mask by keeping depth within a band of the box's median
    depth. No SAM2, no GPU — coarse, but it makes DinoWindowSegmentor runnable
    with zero segmentation checkpoints."""

    source = 'depth-box'

    def __init__(self, band: tuple = (0.8, 1.2), kernel: int = 7,
                 min_valid: int = 10):
        self.band = band
        self.kernel = kernel
        self.min_valid = min_valid

    def masks_for(self, scene: Scene, boxes) -> list[Optional[np.ndarray]]:
        import cv2
        depth = scene.depth
        if depth is None:
            return [None] * len(boxes)
        h, w = depth.shape
        k = np.ones((self.kernel, self.kernel), np.uint8)
        lo_f, hi_f = self.band
        out = []
        for (y0, x0, y1, x1) in boxes:
            roi = depth[y0:y1, x0:x1]
            valid = roi[roi > 0]
            if len(valid) < self.min_valid:
                out.append(None)
                continue
            med = float(np.median(valid))
            band = (depth >= med * lo_f) & (depth <= med * hi_f)
            m = np.zeros((h, w), dtype=bool)
            m[y0:y1, x0:x1] = band[y0:y1, x0:x1]
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k)
            out.append(m.astype(bool))
        return out


# ── Segmentors ──────────────────────────────────────────────────────────

class CNOSSegmentor:
    """Proper CNOS: SAM2 AMG proposes instances, DINOv2 re-ranks each crop
    against the CAD templates. `score` is a cosine similarity in [-1, 1].

    Requires sam2 + checkpoint; raises SegmentorUnavailable without them — it
    does NOT quietly degrade. Put DinoWindowSegmentor after it in a
    FirstAvailableSegmentor chain if you want a SAM2-free fallback."""

    source = 'cnos'

    def __init__(self, renderer, device: str = 'cuda', n_templates: int = 42,
                 sam_model_size: str = 'small', n_masks: int = 5,
                 conf_threshold: float = 0.3, min_pixels: int = 50,
                 dino: Optional[DinoV2Backbone] = None,
                 bank: Optional[CNOSTemplateBank] = None,
                 sam_ckpt_dir: Optional[str] = None):
        self.device = device
        self.sam_model_size = sam_model_size
        self.n_masks = n_masks
        self.conf_threshold = conf_threshold
        self.min_pixels = min_pixels
        self.sam_ckpt_dir = sam_ckpt_dir
        self.dino = dino or DinoV2Backbone(device)
        self.bank = bank or CNOSTemplateBank(renderer, self.dino, n_templates)
        self._amg = None

    def _load_amg(self):
        if self._amg is None:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            model = build_sam2_model(self.sam_model_size, self.device,
                                     self.sam_ckpt_dir)
            self._amg = SAM2AutomaticMaskGenerator(
                model, min_mask_region_area=self.min_pixels, **AMG_PARAMS)
            print(f"SAM2.1-{self.sam_model_size} AMG loaded.")
        return self._amg

    def precompute_templates(self, obj: ObjectModel) -> None:
        """Optional warm-up — segment() builds the bank on first use anyway.
        Call it to fail fast (and to keep render cost out of the timed loop)."""
        self.bank.feats_for(obj)

    def segment(self, scene: Scene, obj: ObjectModel) -> list[Detection]:
        amg = self._load_amg()                  # raises SegmentorUnavailable
        templates = self.bank.feats_for(obj)    # (N_t, D), L2-normed

        dets = []
        for p in amg.generate(scene.rgb):
            m = p['segmentation'].astype(bool)
            if m.sum() < self.min_pixels:
                continue
            crop = _masked_square_crop(scene.rgb, m)
            if crop is None:
                continue
            feat = _l2norm(self.dino.cls_token(crop))
            sim = float((templates @ feat).max())
            if sim < self.conf_threshold:
                continue
            dets.append(Detection(mask=m, score=sim, descriptor=feat,
                                  source=self.source))
        dets.sort(key=lambda d: -d.score)
        return dets[: self.n_masks]


class DinoWindowSegmentor:
    """DINOv2 sliding-window matching: score multi-scale windows over the scene's
    patch-feature grid against the CAD templates, keep the top boxes after NMS,
    then convert each box to a mask with `masker`. `score` is a cosine
    similarity in [-1, 1] — the SAME semantics as CNOSSegmentor, so the two are
    at least comparable, unlike the depth blobs the old code mixed in.

    Coarser than CNOSSegmentor (axis-aligned boxes quantised to 14-px patches,
    no instance-level proposals) but with DepthBoxMasker it needs no SAM2 at
    all. Was `CNOSSegmentor._segment_v0`."""

    source = 'dino-window'

    def __init__(self, renderer, masker, device: str = 'cuda',
                 n_templates: int = 42, n_masks: int = 5,
                 conf_threshold: float = 0.3, min_pixels: int = 200,
                 nms_iou: float = 0.5, dino: Optional[DinoV2Backbone] = None,
                 bank: Optional[CNOSTemplateBank] = None):
        self.masker = masker
        self.n_masks = n_masks
        self.conf_threshold = conf_threshold
        self.min_pixels = min_pixels
        self.nms_iou = nms_iou
        self.dino = dino or DinoV2Backbone(device)
        self.bank = bank or CNOSTemplateBank(renderer, self.dino, n_templates)

    @property
    def source_chain(self) -> str:
        return f"{self.source}+{getattr(self.masker, 'source', '?')}"

    def precompute_templates(self, obj: ObjectModel) -> None:
        self.bank.feats_for(obj)

    def segment(self, scene: Scene, obj: ObjectModel) -> list[Detection]:
        templates = self.bank.feats_for(obj)             # (N_t, D), L2-normed
        feat_map = self.dino.patch_tokens(scene.rgb)     # (n_ph, n_pw, D)
        h_img, w_img = scene.rgb.shape[:2]

        boxes = self._top_boxes(feat_map, templates, h_img, w_img,
                                self.n_masks * 3)
        boxes = [(b, s) for b, s in boxes if s >= self.conf_threshold]
        if not boxes:
            return []

        masks = self.masker.masks_for(scene, [b for b, _ in boxes])
        dets = []
        for (box, score), mask in zip(boxes, masks):
            if mask is None or mask.sum() < self.min_pixels:
                continue
            y0, x0, y1, x1 = box
            dets.append(Detection(mask=mask, score=float(score),
                                  bbox=(x0, y0, x1, y1),
                                  source=self.source_chain))
        dets.sort(key=lambda d: -d.score)
        return dets[: self.n_masks]

    def _top_boxes(self, feat_map: np.ndarray, templates: np.ndarray,
                   h_img: int, w_img: int, n: int) -> list:
        """Multi-scale sliding windows over the patch grid, scored by max cosine
        similarity to any template, then greedy NMS. Returns [((y0,x0,y1,x1), score)]
        in IMAGE coordinates."""
        n_ph, n_pw, D = feat_map.shape
        win_sizes = [(max(1, n_ph // d), max(1, n_pw // d)) for d in (4, 3, 2)]

        proposals = []
        for wh, ww in win_sizes:
            for r in range(0, n_ph - wh + 1, max(1, wh // 2)):
                for c in range(0, n_pw - ww + 1, max(1, ww // 2)):
                    win = feat_map[r:r + wh, c:c + ww].reshape(-1, D).mean(0)
                    score = float((templates @ _l2norm(win)).max())
                    proposals.append(((int(r * h_img / n_ph), int(c * w_img / n_pw),
                                       int((r + wh) * h_img / n_ph),
                                       int((c + ww) * w_img / n_pw)), score))

        proposals.sort(key=lambda x: -x[1])
        kept: list = []
        for box, score in proposals:
            if all(_box_iou(box, k[0]) <= self.nms_iou for k in kept):
                kept.append((box, score))
            if len(kept) >= n:
                break
        return kept


def _box_iou(a, b) -> float:
    y0, x0 = max(a[0], b[0]), max(a[1], b[1])
    y1, x1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, y1 - y0) * max(0, x1 - x0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-8)
