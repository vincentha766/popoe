"""Self-developed CNOS-v3 segmentor.

This is the lab/real-scene track, deliberately separate from official CNOS:

* `source='cnos'` is reserved for the official producer / public BOP files.
* `source='cnos-v3'` means the local recipe: proposal masks -> depth size gate
  -> DINOv2 foreground-patch ranking.

The core gate and patch scoring are numpy-only and unit-testable. Heavy
components (SAM2 proposals, DINOv2 patch extraction, template image loading)
are lazy and optional.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from popoe.interfaces import Detection, ObjectModel, Scene
from popoe.segmentor import AMG_PARAMS, SegmentorUnavailable, build_sam2_model


CNOS_V3_SOURCE = "cnos-v3"
GRID = 16


def _as_K(K) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64)
    if K.shape == (3, 3):
        return K
    if K.size != 9:
        raise ValueError(f"K must be 3x3 or flat length-9, got {K.shape}")
    return K.reshape(3, 3)


def square_crop(img: np.ndarray, mask: np.ndarray,
                *,
                pad: float = 0.1,
                size: int = 224) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Square crop around a mask, resized to `size`.

    The resize uses Pillow to avoid making cv2 a hard import for unit tests.
    """

    ys, xs = np.where(mask)
    if len(ys) < 10:
        return None, None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = max(y1 - y0, x1 - x0)
    half = int(round(side * (0.5 + pad)))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    h_img, w_img = img.shape[:2]
    Y0, Y1 = max(0, cy - half), min(h_img, cy + half)
    X0, X1 = max(0, cx - half), min(w_img, cx + half)
    if Y1 <= Y0 or X1 <= X0:
        return None, None
    try:
        from PIL import Image
    except ImportError as e:
        raise SegmentorUnavailable("Pillow is needed for CNOS-v3 crops") from e

    crop = np.asarray(Image.fromarray(img[Y0:Y1, X0:X1]).resize((size, size)))
    crop_mask = np.asarray(
        Image.fromarray(mask[Y0:Y1, X0:X1].astype(np.uint8)).resize(
            (size, size), resample=Image.Resampling.NEAREST
        )
    ).astype(bool)
    return crop, crop_mask


def _l2norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


def _bbox_xyxy(mask: np.ndarray) -> Optional[tuple]:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


@dataclass(frozen=True)
class DepthSizeGate:
    """Depth-based 3D extent gate for proposal masks.

    The extent is the percentile 3D bbox diagonal of mask pixels with valid
    depth, after dropping depth values outside a band around the mask median.
    The accepted interval is expressed as a fraction of `ObjectModel.diameter`.
    """

    min_extent_ratio: float = 0.25
    max_extent_ratio: float = 1.1
    depth_band_m: float = 0.10
    min_depth_m: float = 0.05
    min_points: int = 50
    min_pixels: int = 8000
    percentiles: tuple[float, float] = (2.0, 98.0)

    def extent_3d(self, mask: np.ndarray, depth_m: np.ndarray, K) -> Optional[float]:
        K = _as_K(K)
        valid = mask.astype(bool) & (depth_m > self.min_depth_m)
        ys, xs = np.where(valid)
        if len(ys) < self.min_points:
            return None
        d = depth_m[ys, xs].astype(np.float64)
        med = float(np.median(d))
        band = np.abs(d - med) < self.depth_band_m
        if int(band.sum()) < self.min_points:
            return None
        ys, xs, d = ys[band], xs[band], d[band]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        X = (xs.astype(np.float64) - cx) * d / fx
        Y = (ys.astype(np.float64) - cy) * d / fy
        P = np.stack([X, Y, d], axis=1)
        lo, hi = np.percentile(P, self.percentiles, axis=0)
        return float(np.linalg.norm(hi - lo))

    def accepts(self, scene: Scene, obj: ObjectModel,
                mask: np.ndarray) -> tuple[bool, Optional[float]]:
        if scene.depth is None or scene.K is None:
            return False, None
        if int(mask.sum()) < self.min_pixels:
            return False, None
        extent = self.extent_3d(mask, scene.depth, scene.K)
        if extent is None:
            return False, None
        lo = self.min_extent_ratio * float(obj.diameter)
        hi = self.max_extent_ratio * float(obj.diameter)
        return lo <= extent <= hi, extent


class PatchForegroundScorer:
    """Foreground patch ranking: query patches vs template foreground patches."""

    def __init__(self, extractor=None, topk: int = 40,
                 grid: int = GRID, fg_threshold: float = 0.4):
        self.extractor = extractor
        self.topk = int(topk)
        self.grid = int(grid)
        self.fg_threshold = float(fg_threshold)

    def foreground_patch_mask(self, fg_mask: np.ndarray) -> np.ndarray:
        h, w = fg_mask.shape
        if h % self.grid != 0 or w % self.grid != 0:
            raise ValueError(
                f"foreground mask shape {fg_mask.shape} must be divisible by {self.grid}"
            )
        cell_h, cell_w = h // self.grid, w // self.grid
        occ = fg_mask.reshape(self.grid, cell_h, self.grid, cell_w).mean(axis=(1, 3))
        keep = occ.reshape(-1) > self.fg_threshold
        if int(keep.sum()) < 3:
            keep[:] = True
        return keep

    def score_tokens(self, query_tokens: np.ndarray,
                     template_tokens: np.ndarray) -> float:
        q = _l2norm(np.asarray(query_tokens, dtype=np.float64), axis=1)
        t = _l2norm(np.asarray(template_tokens, dtype=np.float64), axis=1)
        if q.size == 0 or t.size == 0:
            return 0.0
        best = (q @ t.T).max(axis=1)
        k = min(self.topk, best.size)
        if k <= 0:
            return 0.0
        top = np.partition(best, best.size - k)[best.size - k:]
        return float(top.mean())

    def score_mask(self, rgb: np.ndarray, mask: np.ndarray,
                   template_tokens: np.ndarray) -> float:
        if self.extractor is None:
            raise SegmentorUnavailable("CNOS-v3 patch scorer needs a DINOv2 extractor")
        crop, crop_mask = square_crop(rgb, mask)
        if crop is None:
            return 0.0
        q = self.extractor.fg_patches(crop, crop_mask)
        return self.score_tokens(q, template_tokens)


class DinoV2ForegroundPatchExtractor:
    """Lazy DINOv2 patch extractor matching `gedi/scripts/cnos_match3.py`."""

    def __init__(self, device: str = "cuda", model_name: str = "dinov2_vitg14_reg",
                 grid: int = GRID):
        self.device = device
        self.model_name = model_name
        self.grid = int(grid)
        self._model = None
        self._mean = None
        self._std = None

    @property
    def model(self):
        if self._model is None:
            try:
                import torch
            except ImportError as e:
                raise SegmentorUnavailable(f"torch not installed: {e}") from e
            try:
                self._model = torch.hub.load(
                    "facebookresearch/dinov2", self.model_name, pretrained=True
                ).to(self.device).eval()
            except Exception as e:
                raise SegmentorUnavailable(
                    f"DINOv2 ({self.model_name}) load failed: {e}"
                ) from e
            self._mean = torch.tensor(
                [0.485, 0.456, 0.406], device=self.device
            ).view(1, 3, 1, 1)
            self._std = torch.tensor(
                [0.229, 0.224, 0.225], device=self.device
            ).view(1, 3, 1, 1)
        return self._model

    def fg_patches(self, rgb_uint8: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
        import torch

        model = self.model
        x = torch.from_numpy(rgb_uint8).float().permute(2, 0, 1).unsqueeze(0)
        x = x.to(self.device) / 255.0
        x = (x - self._mean) / self._std
        with torch.no_grad():
            tok = model.forward_features(x)["x_norm_patchtokens"][0]
        keep = PatchForegroundScorer(grid=self.grid).foreground_patch_mask(fg_mask)
        tok = tok[torch.from_numpy(keep).to(self.device)]
        return torch.nn.functional.normalize(tok, dim=1).detach().cpu().numpy()


class TemplateDirPatchBank:
    """Patch-token bank over rendered template PNGs.

    `template_dir` may be a string, a `{obj_id: dir}` mapping, or a callable
    `ObjectModel -> dir`.
    """

    def __init__(self, template_dir: str | Mapping[int, str] | Callable[[ObjectModel], str],
                 extractor: DinoV2ForegroundPatchExtractor,
                 scorer: Optional[PatchForegroundScorer] = None):
        self.template_dir = template_dir
        self.extractor = extractor
        self.scorer = scorer or PatchForegroundScorer(extractor=extractor)
        self._cache: dict[tuple[int, str], np.ndarray] = {}

    def _dir_for(self, obj: ObjectModel) -> str:
        if callable(self.template_dir):
            return str(self.template_dir(obj))
        if isinstance(self.template_dir, MappingABC):
            return str(self.template_dir[obj.obj_id])
        return str(self.template_dir)

    def patches_for(self, obj: ObjectModel) -> np.ndarray:
        key = (obj.obj_id, self._dir_for(obj))
        if key in self._cache:
            return self._cache[key]
        try:
            from PIL import Image
        except ImportError as e:
            raise SegmentorUnavailable("Pillow is needed to load CNOS-v3 templates") from e

        toks = []
        for path in sorted(glob.glob(os.path.join(key[1], "*.png"))):
            im = np.asarray(Image.open(path))
            if im.ndim != 3:
                continue
            if im.shape[2] == 4:
                fg = im[..., 3] > 0
                rgb = im[..., :3]
            else:
                rgb = im[..., :3]
                fg = np.any(rgb > 12, axis=2)
            crop, cmask = square_crop(rgb.astype(np.uint8), fg)
            if crop is not None:
                toks.append(self.extractor.fg_patches(crop, cmask))
        if not toks:
            raise SegmentorUnavailable(f"no usable CNOS-v3 templates in {key[1]}")
        bank = np.concatenate(toks, axis=0)
        self._cache[key] = bank
        return bank


class SAM2AMGMaskProposer:
    """SAM2 AMG mask proposer for CNOS-v3."""

    source = "sam2-amg"

    def __init__(self, device: str = "cuda", model_size: str = "small",
                 min_mask_region_area: int = 2000,
                 sam_ckpt_dir: Optional[str] = None):
        self.device = device
        self.model_size = model_size
        self.min_mask_region_area = int(min_mask_region_area)
        self.sam_ckpt_dir = sam_ckpt_dir
        self._generator = None

    def _load(self):
        if self._generator is None:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

            model = build_sam2_model(self.model_size, self.device, self.sam_ckpt_dir)
            self._generator = SAM2AutomaticMaskGenerator(
                model, min_mask_region_area=self.min_mask_region_area, **AMG_PARAMS
            )
        return self._generator

    def propose(self, scene: Scene) -> list[np.ndarray]:
        gen = self._load()
        return [p["segmentation"].astype(bool) for p in gen.generate(scene.rgb)]


class CNOSv3Segmentor:
    """Depth-size-gated foreground-patch CNOS segmentor."""

    source = CNOS_V3_SOURCE

    def __init__(self, proposer=None, template_bank=None,
                 scorer: Optional[PatchForegroundScorer] = None,
                 size_gate: Optional[DepthSizeGate] = None,
                 n_masks: int = 5,
                 conf_threshold: float = -1.0):
        self.proposer = proposer
        self.template_bank = template_bank
        self.scorer = scorer or PatchForegroundScorer()
        self.size_gate = size_gate or DepthSizeGate()
        self.n_masks = int(n_masks)
        self.conf_threshold = float(conf_threshold)

    def segment(self, scene: Scene, obj: ObjectModel) -> list[Detection]:
        if self.proposer is None:
            raise SegmentorUnavailable("CNOS-v3 needs a mask proposer")
        if self.template_bank is None:
            raise SegmentorUnavailable("CNOS-v3 needs a template patch bank")
        template_tokens = self.template_bank.patches_for(obj)
        dets: list[Detection] = []
        for mask in self.proposer.propose(scene):
            mask = mask.astype(bool)
            ok, extent = self.size_gate.accepts(scene, obj, mask)
            if not ok:
                continue
            score = self.scorer.score_mask(scene.rgb, mask, template_tokens)
            if score < self.conf_threshold:
                continue
            dets.append(Detection(
                mask=mask,
                score=float(score),
                bbox=_bbox_xyxy(mask),
                source=self.source,
                descriptor=np.asarray([extent], dtype=np.float32),
            ))
        dets.sort(key=lambda d: -d.score)
        return dets[: self.n_masks]


__all__ = [
    "CNOSv3Segmentor",
    "DepthSizeGate",
    "DinoV2ForegroundPatchExtractor",
    "PatchForegroundScorer",
    "SAM2AMGMaskProposer",
    "TemplateDirPatchBank",
    "square_crop",
]
