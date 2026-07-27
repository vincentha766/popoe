"""Self-developed CNOS-lab segmentor.

This is the lab/real-scene track, deliberately separate from official CNOS:

* `source='cnos'` is reserved for the official producer / public BOP files.
* `source='cnos-lab'` means the local recipe: proposal masks -> depth size gate
  -> DINOv2 foreground-patch ranking.

Formerly named ``cnos-v3``: the "v3" was the iteration count of gedi's
``cnos_match3.py`` lab script, not an official CNOS release — renamed because
outside readers had no way to know that. ``popoe.segmentor_cnos_v3`` remains as
an import shim; old artifacts carrying ``source="cnos-v3"`` mean this recipe.

The core gate and patch scoring are numpy-only and unit-testable. Heavy
components (SAM2 proposals, DINOv2 patch extraction, template image loading)
are lazy and optional.
"""

from __future__ import annotations

import glob
import math
import os
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from popoe.interfaces import Detection, ObjectModel, Scene, is_runtime_failure
from popoe.segmentor import AMG_PARAMS, SegmentorUnavailable, build_sam2_model


CNOS_LAB_SOURCE = "cnos-lab"
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
        raise SegmentorUnavailable("Pillow is needed for CNOS-lab crops") from e

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

    Set ``enabled=False`` to disable the band (and min-pixel) filter while still
    exposing ``extent_3d`` for diagnostics / descriptors (G1 paper-mode A/B).
    """

    min_extent_ratio: float = 0.25
    max_extent_ratio: float = 1.1
    depth_band_m: float = 0.10
    min_depth_m: float = 0.05
    min_points: int = 50
    min_pixels: int = 8000
    percentiles: tuple[float, float] = (2.0, 98.0)
    enabled: bool = True

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
        if int(mask.sum()) == 0:
            return False, None
        if not self.enabled:
            extent = None
            if scene.depth is not None and scene.K is not None:
                extent = self.extent_3d(mask, scene.depth, scene.K)
            return True, extent
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


@dataclass(frozen=True)
class DiameterSizeModel:
    """Compare a measured 3D extent to one or more CAD diameters.

    The band gate in :class:`DepthSizeGate` is a *one-sided* filter: a small
    confuser still sits inside a larger object's ``[0.25, 1.1]×diam`` band, so
    it cannot disambiguate YCB-V clamps when the query is the larger one.
    This model instead scores **relative** size fit (log-ratio), so each mask
    prefers the diameter it is nearest to — the missing half of size-aware
    mask selection. Registration-stage bidirectional ``metric_fit``
    (:class:`popoe.scoring.ChampionScorer`) remains complementary: it needs a
    pose; this runs on depth alone.
    """

    # ~0.2 separates YCB-V clamp diameters under competitive softmax; 0.35 is too soft
    # to overturn large wrong-label CNOS score gaps (measured in size-select A/B).
    sigma_log: float = 0.20

    def log_ratio_error(self, extent: float, diameter: float) -> float:
        e = max(float(extent), 1e-9)
        d = max(float(diameter), 1e-9)
        return abs(math.log(e / d))

    def relative_error(self, extent: float, diameter: float) -> float:
        d = max(float(diameter), 1e-9)
        return abs(float(extent) - d) / d

    def affinity(self, extent: float, diameter: float) -> float:
        """Soft size weight in ``(0, 1]``; 1 when extent == diameter."""
        err = self.log_ratio_error(extent, diameter)
        sig = max(float(self.sigma_log), 1e-6)
        return float(math.exp(-0.5 * (err / sig) ** 2))

    def nearest_obj_id(self, extent: float,
                      diameters: Mapping[int, float]) -> int:
        if not diameters:
            raise ValueError("diameters mapping is empty")
        return min(
            diameters.keys(),
            key=lambda oid: self.log_ratio_error(extent, diameters[oid]),
        )

    def soft_score(self, appearance: float, extent: Optional[float],
                   diameter: float, *,
                   missing_extent_affinity: float = 0.0) -> float:
        """Appearance × size affinity; missing extent → ``missing_extent_affinity``."""
        if extent is None:
            return float(appearance) * float(missing_extent_affinity)
        return float(appearance) * self.affinity(extent, diameter)


@dataclass(frozen=True)
class SizeSelectResult:
    """One selected candidate after size-aware ranking."""
    index: int
    appearance: float
    extent: Optional[float]
    size_score: float
    assigned_obj_id: Optional[int]


def select_by_soft_affinity(
    appearances: Sequence[float],
    extents: Sequence[Optional[float]],
    diameter: float,
    model: Optional[DiameterSizeModel] = None,
    *,
    missing_extent_affinity: float = 0.0,
    rival_diameters: Optional[Sequence[float]] = None,
) -> Optional[SizeSelectResult]:
    """Pick argmax_i appearance_i × size weight for ``diameter``.

    If ``rival_diameters`` is set (other CADs in a confusable set), the size
    weight is the *softmax share* of affinity for the query diameter among
    {query}∪rivals. That competes sizes fairly when appearance margins are
    large (band/Gaussian alone often cannot overturn a strong wrong-label
    CNOS score). Without rivals, weight is the raw Gaussian affinity.
    """
    if len(appearances) != len(extents):
        raise ValueError("appearances and extents length mismatch")
    if not appearances:
        return None
    model = model or DiameterSizeModel()
    rivals = list(rival_diameters or ())
    best_i = -1
    best_s = -1.0
    for i, (app, ext) in enumerate(zip(appearances, extents)):
        if ext is None:
            w = float(missing_extent_affinity)
        elif rivals:
            aff_q = model.affinity(ext, diameter)
            aff_sum = aff_q + sum(model.affinity(ext, rd) for rd in rivals)
            w = aff_q / aff_sum if aff_sum > 0 else 0.0
        else:
            w = model.affinity(ext, diameter)
        s = float(app) * w
        if s > best_s:
            best_s = s
            best_i = i
    if best_i < 0:
        return None
    return SizeSelectResult(
        index=best_i,
        appearance=float(appearances[best_i]),
        extent=extents[best_i],
        size_score=float(best_s),
        assigned_obj_id=None,
    )


def select_by_nearest_diameter(
    appearances: Sequence[float],
    extents: Sequence[Optional[float]],
    query_obj_id: int,
    diameters: Mapping[int, float],
    model: Optional[DiameterSizeModel] = None,
    *,
    fallback_appearance: bool = True,
) -> Optional[SizeSelectResult]:
    """Keep only masks whose nearest CAD diameter is ``query_obj_id``, then top appearance.

    If none remain and ``fallback_appearance`` is True, fall back to pure
    appearance top-1 (same practical default as band-gate empty handling).
    """
    if len(appearances) != len(extents):
        raise ValueError("appearances and extents length mismatch")
    if not appearances:
        return None
    if query_obj_id not in diameters:
        raise KeyError(f"query_obj_id {query_obj_id} missing from diameters")
    model = model or DiameterSizeModel()
    kept: list[tuple[int, float, float]] = []  # i, appearance, extent
    for i, (app, ext) in enumerate(zip(appearances, extents)):
        if ext is None:
            continue
        if model.nearest_obj_id(ext, diameters) == query_obj_id:
            kept.append((i, float(app), float(ext)))
    if kept:
        i, app, ext = max(kept, key=lambda t: t[1])
        return SizeSelectResult(
            index=i,
            appearance=app,
            extent=ext,
            size_score=app,
            assigned_obj_id=query_obj_id,
        )
    if not fallback_appearance:
        return None
    i = max(range(len(appearances)), key=lambda j: float(appearances[j]))
    return SizeSelectResult(
        index=i,
        appearance=float(appearances[i]),
        extent=extents[i],
        size_score=float(appearances[i]),
        assigned_obj_id=None,
    )


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
            raise SegmentorUnavailable("CNOS-lab patch scorer needs a DINOv2 extractor")
        crop, crop_mask = square_crop(rgb, mask)
        if crop is None:
            return 0.0
        q = self.extractor.fg_patches(crop, crop_mask)
        return self.score_tokens(q, template_tokens)


class DinoV2ForegroundPatchExtractor:
    """Lazy DINOv2 patch extractor matching `gedi/scripts/cnos_match3.py`."""

    def __init__(self, device: str = "cuda", model_name: str = "dinov2_vitg14_reg",
                 grid: int = GRID, model=None):
        # model: preloaded DINOv2 module — the sharing seam (popoe.assembly).
        # Must be the weights model_name names, on `device`.
        self.device = device
        self.model_name = model_name
        self.grid = int(grid)
        self._model = model
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
                if is_runtime_failure(e):    # an OOM is not "backend missing"
                    raise
                raise SegmentorUnavailable(
                    f"DINOv2 ({self.model_name}) load failed: {e}"
                ) from e
        return self._model

    def _ensure_norm_stats(self):
        # Split out of the loader so an injected model gets them too.
        if self._mean is None:
            import torch
            self._mean = torch.tensor(
                [0.485, 0.456, 0.406], device=self.device
            ).view(1, 3, 1, 1)
            self._std = torch.tensor(
                [0.229, 0.224, 0.225], device=self.device
            ).view(1, 3, 1, 1)

    def fg_patches(self, rgb_uint8: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
        import torch

        model = self.model
        self._ensure_norm_stats()
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
            raise SegmentorUnavailable("Pillow is needed to load CNOS-lab templates") from e

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
            raise SegmentorUnavailable(f"no usable CNOS-lab templates in {key[1]}")
        bank = np.concatenate(toks, axis=0)
        self._cache[key] = bank
        return bank


class SAM2AMGMaskProposer:
    """SAM2 AMG mask proposer for CNOS-lab."""

    source = "sam2-amg"

    def __init__(self, device: str = "cuda", model_size: str = "small",
                 min_mask_region_area: int = 2000,
                 sam_ckpt_dir: Optional[str] = None,
                 sam_model=None):
        # sam_model: prebuilt SAM2 model — the sharing seam (popoe.assembly).
        # Must be the checkpoint model_size names.
        self.device = device
        self.model_size = model_size
        self.min_mask_region_area = int(min_mask_region_area)
        self.sam_ckpt_dir = sam_ckpt_dir
        self._sam_model = sam_model
        self._generator = None

    def _load(self):
        if self._generator is None:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

            model = (self._sam_model if self._sam_model is not None else
                     build_sam2_model(self.model_size, self.device, self.sam_ckpt_dir))
            self._generator = SAM2AutomaticMaskGenerator(
                model, min_mask_region_area=self.min_mask_region_area, **AMG_PARAMS
            )
        return self._generator

    def propose(self, scene: Scene) -> list[np.ndarray]:
        gen = self._load()
        return [p["segmentation"].astype(bool) for p in gen.generate(scene.rgb)]


class CNOSLabSegmentor:
    """Depth-size-gated foreground-patch CNOS segmentor."""

    source = CNOS_LAB_SOURCE

    def __init__(self, proposer=None, template_bank=None,
                 scorer: Optional[PatchForegroundScorer] = None,
                 size_gate: Optional[DepthSizeGate] = None,
                 n_masks: int = 5,
                 conf_threshold: float = -1.0):
        self.proposer = proposer
        self.template_bank = template_bank
        if scorer is None and template_bank is not None:
            scorer = getattr(template_bank, "scorer", None)
            extractor = getattr(template_bank, "extractor", None)
            if scorer is None and extractor is not None:
                scorer = PatchForegroundScorer(extractor=extractor)
        self.scorer = scorer or PatchForegroundScorer()
        self.size_gate = size_gate or DepthSizeGate()
        self.n_masks = int(n_masks)
        self.conf_threshold = float(conf_threshold)

    def segment(self, scene: Scene, obj: ObjectModel) -> list[Detection]:
        if self.proposer is None:
            raise SegmentorUnavailable("CNOS-lab needs a mask proposer")
        if self.template_bank is None:
            raise SegmentorUnavailable("CNOS-lab needs a template patch bank")
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
    "CNOSLabSegmentor",
    "DepthSizeGate",
    "DiameterSizeModel",
    "DinoV2ForegroundPatchExtractor",
    "PatchForegroundScorer",
    "SAM2AMGMaskProposer",
    "SizeSelectResult",
    "TemplateDirPatchBank",
    "select_by_nearest_diameter",
    "select_by_soft_affinity",
    "square_crop",
]
