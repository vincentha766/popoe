"""MUSE-style segmentor — a reimplementation, and its detections producer.

MUSE (arXiv 2510.17866, Cho/Park/Oh) is one of the four mask sources FreeZeV2
ensembles, and the only one with **no public code and no downloadable masks**
(BOP method_info/873). So unlike CNOS / SAM-6D / NIDS there is no official
producer to adapt: what popoe can offer is a reimplementation from the paper,
and that distinction is part of the contract, not a footnote.

Naming (same discipline as CNOS's ``cnos`` / ``cnos-v3`` / ``cnos-live`` split):

============  ==============================================================
``muse``      RESERVED for official MUSE artefacts. Nothing here writes it.
``muse-repro``This module: our from-the-paper reimplementation.
============  ==============================================================

Do not relabel this output as ``muse``. The reproduction study cites MUSE as
evidence that one of the four ensemble members is externally unreproducible; a
number produced here wearing the official name would quietly refute an argument
we are making on purpose.

Two halves, because MUSE has to be both kinds of segmentor at once
(see ARCHITECTURE.md, "File-based detection backends"):

* **live** — ``MuseSegmentor`` computes masks from pixels, like
  ``CNOSv3Segmentor``. This is what runs on a freshly captured frame.
* **producer** — ``muse_records`` / ``write_muse_detections`` dump the same
  masks to a BOP-format detections JSON, so the result becomes a file-backed
  source like every other backend: it can join a ``BOPDetectionsSegmentor``
  union, be scored by ``popoe.segmentation_eval``, and be archived alongside
  the run that produced it.

Pipeline (paper Fig.2 / Eq.1-10):

1. Onboarding — per class, a template bank of (class-token, GeM patch) pairs
   over rendered PNGs.
2. Proposal — Grounding DINO with a generic text prompt gives boxes and an
   objectness P(O|p); SAM2 box-prompting turns each box into a mask.
3. Embedding — DINOv2 gives the class token and patch tokens; patch tokens are
   foreground-masked then GeM-pooled.
   ``S_abs(p,c) = max_t [ alpha*cos(cls_p, cls_t) + (1-alpha)*Tanimoto(gem_p, gem_t) ]``
4. Matching — ``S_rel = softmax_c(S_abs/tau)`` over ALL registered classes,
   ``S_joint = beta*S_abs + (1-beta)*S_rel``, ``S_final = P(O|p)^gamma * S_joint``.

Deliberate divergences from the paper, carried over from the validated script
``gedi/scripts/muse_match.py`` (keep this list honest — it is what stops these
numbers being read as an exact replication):

* A depth-based 3D-extent size gate runs after proposal. MUSE has none (its BOP
  results are RGB-only); we keep it because it is cheap and directly targets the
  printed-text confuser failure mode seen on this project's real shots.
* No union-bbox box-prompt refinement pass (CNOS-v3 has one); the GD+SAM2
  cascade is the paper's single largest ablation lever and should not need it.
* Embedding fusion is simplified: the paper's vMF-weighted multi-embedding
  similarity is replaced by GeM pooling + Tanimoto, which needs no per-class
  concentration parameters fitted from a template set this small.

The scoring core is numpy-only and unit-tested; every heavy component (Grounding
DINO, SAM2, DINOv2, template PNGs) is lazy, injected, and raises
``SegmentorUnavailable`` rather than degrading into a different method.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from popoe.cache import fingerprint
from popoe.interfaces import Detection, ObjectModel, Scene
from popoe.segmentor import SegmentorUnavailable, build_sam2_model
from popoe.segmentor_detections import BOPDetectionsSegmentor
from popoe.segmentor_cnos_v3 import (
    GRID,
    PatchForegroundScorer,
    DepthSizeGate,
    _bbox_xyxy,          # package-internal; shared so the bbox convention cannot drift
    square_crop,
)

MUSE_SOURCE = "muse-repro"

#: The name this module must never write. `muse` belongs to official MUSE
#: artefacts, which do not exist publicly; a reimplementation wearing it would
#: refute the very claim the reproduction study makes about MUSE.
RESERVED_SOURCE = "muse"

#: Layout of ``Detection.descriptor`` for this segmentor. The final score alone
#: cannot say WHY a mask ranked where it did (absolute match? objectness? size?),
#: and the ablations need that breakdown offline.
DESCRIPTOR_FIELDS = ("s_abs", "s_rel", "p_obj", "extent_m")

# Paper/script defaults, kept in one place so the CLI, the class and the docs
# cannot drift apart.
DEFAULT_ALPHA = 0.5      # class-token vs patch-token weight in S_abs
DEFAULT_BETA = 0.8       # absolute vs relative weight in S_joint
DEFAULT_TAU = 0.02       # relative-score softmax temperature
DEFAULT_GAMMA = 0.1      # objectness confidence exponent
DEFAULT_GEM_P = 1.5      # GeM pooling exponent (Radenovic et al.)
DEFAULT_PROMPT = "items."
DEFAULT_GDINO_MODEL = "IDEA-Research/grounding-dino-base"
#: MUSE keeps small proposals CNOS-v3 would drop, so the gate's pixel floor is
#: lower here than that segmentor's 8000.
DEFAULT_MIN_PIXELS = 2000


# ── Scoring core (numpy only — no torch, no GPU, unit-tested) ────────────

def gem_pool(tokens: np.ndarray, p: float = DEFAULT_GEM_P,
             eps: float = 1e-6) -> np.ndarray:
    """Generalised-mean pooling over the token axis: ``(L, d) -> (d,)``.

    Tokens are clamped at `eps` before the fractional power, because DINOv2
    patch tokens are signed and a negative base with a fractional exponent is
    NaN. This matches the reference implementation.
    """
    t = np.asarray(tokens, dtype=np.float64)
    if t.ndim != 2:
        raise ValueError(f"gem_pool expects (L, d) tokens, got {t.shape}")
    if t.shape[0] == 0:
        raise ValueError("gem_pool got zero tokens")
    return np.power(np.mean(np.power(np.clip(t, eps, None), p), axis=0), 1.0 / p)


def tanimoto(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """Continuous Tanimoto (generalised Jaccard) similarity of two vectors."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    dot = float(a @ b)
    return float(dot / (float(a @ a) + float(b @ b) - dot + eps))


def _component_identity(component) -> dict:
    """A hashable summary of one injected component's settings.

    A component may declare its own identity with a ``config()`` method — the
    escape hatch for anything holding public MUTABLE state (a request counter,
    a handle) that must not enter the key. Otherwise this reflects over public
    scalar attributes: model ids, thresholds, exponents. Loaded models and
    caches live under `_`-prefixed names and are skipped, so this stays cheap
    and stable across reloads. Reflective by default because a hand-written
    list per component is exactly what goes stale when a knob is added.
    """
    if component is None:
        return {"class": None}
    declared = getattr(component, "config", None)
    if callable(declared):
        return {"class": type(component).__name__, **dict(declared())}

    def keep(v):
        if isinstance(v, (str, int, float, bool, tuple, type(None))):
            return True
        values = v.values() if isinstance(v, dict) else v
        return isinstance(v, (dict, list)) and all(
            isinstance(x, (str, int, float, bool, type(None))) for x in values)

    fields = getattr(component, "__dict__", {})
    return {"class": type(component).__name__,
            **{k: v for k, v in sorted(fields.items())
               if not k.startswith("_") and keep(v)}}


def _check_source(source: str) -> str:
    """Refuse the reserved official name at every point that can stamp one.

    Documentation is not enough here: the whole value of the split is that no
    artefact in this repo can claim to be official MUSE, and a `source=` keyword
    is exactly how that would happen by accident.
    """
    if str(source).strip().lower() == RESERVED_SOURCE:
        raise ValueError(
            f"source={source!r} is reserved for official MUSE artefacts, which "
            f"are not public. This reimplementation writes {MUSE_SOURCE!r}.")
    return source


def _check_n_masks(n: int) -> int:
    """Reject a negative mask count instead of letting `[:n]` reinterpret it.

    `dets[:-1]` silently drops the LAST detection rather than returning none,
    so a negative topk would quietly return a truncated ranking that looks
    plausible.
    """
    n = int(n)
    if n < 0:
        raise ValueError(f"n_masks must be >= 0, got {n}")
    return n


def _is_runtime_failure(exc: BaseException) -> bool:
    """True for failures that must PROPAGATE rather than read as unavailability.

    A CUDA OOM during model load is a runtime failure: turning it into
    ``SegmentorUnavailable`` lets a ``FirstAvailableSegmentor`` chain quietly
    continue to a weaker method and buries the real cause (ARCHITECTURE.md,
    "the availability contract"). Matched without importing torch.

    Matching the class name alone is not enough: only torch >= 1.13 raises the
    dedicated ``torch.cuda.OutOfMemoryError``, and plenty of allocation failures
    still surface as a bare ``RuntimeError('CUDA out of memory. Tried to
    allocate ...')`` — the exact shape that must not be mistaken for a missing
    backend.
    """
    if isinstance(exc, MemoryError) or type(exc).__name__ == "OutOfMemoryError":
        return True
    text = str(exc).lower()
    return isinstance(exc, RuntimeError) and (
        "out of memory" in text or "cuda error" in text
        or "cublas_status_alloc_failed" in text)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    return v / (np.linalg.norm(v) + 1e-12)


def absolute_score(cls_q: np.ndarray, gem_q: np.ndarray,
                   cls_bank: np.ndarray, gem_bank: np.ndarray,
                   alpha: float = DEFAULT_ALPHA) -> float:
    """``S_abs(p,c)``: best-matching template view of one class.

    `cls_bank` / `gem_bank` are ``(T, d)`` stacks over that class's templates.
    Class tokens are L2-normalised here so the cosine is well defined whatever
    the caller passed.
    """
    cls_bank = np.atleast_2d(np.asarray(cls_bank, dtype=np.float64))
    gem_bank = np.atleast_2d(np.asarray(gem_bank, dtype=np.float64))
    if cls_bank.shape[0] != gem_bank.shape[0]:
        raise ValueError(
            f"template bank mismatch: {cls_bank.shape[0]} class tokens vs "
            f"{gem_bank.shape[0]} patch embeddings")
    q = _unit(cls_q)
    best = -np.inf
    for cls_t, gem_t in zip(cls_bank, gem_bank):
        s = alpha * float(q @ _unit(cls_t)) + (1.0 - alpha) * tanimoto(gem_q, gem_t)
        best = max(best, s)
    return float(best)


def relative_scores(s_abs: np.ndarray, tau: float = DEFAULT_TAU) -> np.ndarray:
    """``S_rel = softmax_c(S_abs/tau)`` across classes, row-wise on ``(P, C)``.

    Softmax is computed max-subtracted. On any input the reference
    implementation could handle this is identical to it; it additionally keeps
    a row whose candidates all failed (every entry ``-inf``) from becoming NaN,
    which would otherwise propagate into every class's ranking. Such a row is
    uniform — it carries no evidence for any class, and the proposal is dropped
    downstream anyway.
    """
    s = np.asarray(s_abs, dtype=np.float64)
    if s.ndim != 2:
        raise ValueError(f"relative_scores expects (P, C), got {s.shape}")
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    if s.shape[1] == 0:
        return np.zeros_like(s)
    z = s / tau
    peak = z.max(axis=1, keepdims=True)
    z = z - np.where(np.isfinite(peak), peak, 0.0)
    e = np.exp(z)
    total = e.sum(axis=1, keepdims=True)
    return np.where(total > 0, e / np.where(total > 0, total, 1.0), 1.0 / s.shape[1])


def joint_scores(s_abs: np.ndarray, s_rel: np.ndarray,
                 beta: float = DEFAULT_BETA) -> np.ndarray:
    """``S_joint = beta*S_abs + (1-beta)*S_rel``."""
    return beta * np.asarray(s_abs, dtype=np.float64) + \
        (1.0 - beta) * np.asarray(s_rel, dtype=np.float64)


def final_scores(s_joint: np.ndarray, p_obj: np.ndarray,
                 gamma: float = DEFAULT_GAMMA) -> np.ndarray:
    """``S_final = P(O|p)^gamma * S_joint`` — objectness as a confidence prior."""
    s_joint = np.asarray(s_joint, dtype=np.float64)
    p = np.clip(np.asarray(p_obj, dtype=np.float64), 0.0, None).reshape(-1, 1)
    return (p ** gamma) * s_joint


# ── Registered classes ──────────────────────────────────────────────────

@dataclass(frozen=True)
class MuseClass:
    """One object class MUSE can match against: CAD metadata + template dir.

    The `ObjectModel` is needed here (not just at ``segment`` time) because the
    proposal-stage size gate accepts a mask plausible for ANY registered class,
    which requires every class's diameter before matching starts.
    """
    obj: ObjectModel
    template_dir: str


# ── Heavy components (lazy; each raises SegmentorUnavailable) ────────────

class GroundingDinoBoxProposer:
    """Grounding DINO open-vocabulary boxes + objectness, via transformers.

    The generic prompt ("items.") is the paper's: MUSE is category-agnostic at
    proposal time and defers identity entirely to the template matching.
    """

    source = "grounding-dino"

    def __init__(self, model_id: str = DEFAULT_GDINO_MODEL,
                 prompt: str = DEFAULT_PROMPT,
                 box_threshold: float = 0.15,
                 text_threshold: float = 0.15,
                 device: str = "cuda"):
        self.model_id = model_id
        self.prompt = prompt
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.device = device
        self._processor = None
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                import torch  # noqa: F401 — guarded so a missing torch is availability
            except ImportError as e:
                raise SegmentorUnavailable(f"torch not installed: {e}") from e
            try:
                from transformers import (AutoModelForZeroShotObjectDetection,
                                          AutoProcessor)
            except ImportError as e:
                raise SegmentorUnavailable(
                    "transformers not installed: pip install transformers") from e
            try:
                self._processor = AutoProcessor.from_pretrained(self.model_id)
                self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                    self.model_id).to(self.device).eval()
            except Exception as e:
                if _is_runtime_failure(e):
                    raise
                raise SegmentorUnavailable(
                    f"Grounding DINO ({self.model_id}) load failed: {e}") from e
        return self._processor, self._model

    def config(self) -> dict:
        return {"model_id": self.model_id, "prompt": self.prompt,
                "box_threshold": self.box_threshold,
                "text_threshold": self.text_threshold}

    def propose(self, scene: Scene) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(boxes (N,4) xyxy pixels, objectness (N,))``."""
        # _load first: it reports a missing backend as SegmentorUnavailable,
        # whereas a bare `import torch` here would raise ModuleNotFoundError
        # past the fallback chain.
        processor, model = self._load()
        import torch

        try:
            from PIL import Image
        except ImportError as e:
            raise SegmentorUnavailable("Pillow is needed for MUSE proposals") from e
        img = Image.fromarray(np.asarray(scene.rgb, dtype=np.uint8))
        inputs = processor(images=img, text=self.prompt,
                           return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)
        res = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=self.box_threshold, text_threshold=self.text_threshold,
            target_sizes=[(scene.rgb.shape[0], scene.rgb.shape[1])])[0]
        return (res["boxes"].cpu().numpy().astype(np.float64),
                res["scores"].cpu().numpy().astype(np.float64))


class SAM2BoxMaskRefiner:
    """SAM2 box-prompted masks. One ``set_image`` per frame, not per box —
    the image embedding is the expensive half."""

    source = "sam2-box"

    def __init__(self, device: str = "cuda", model_size: str = "large",
                 sam_ckpt_dir: Optional[str] = None):
        self.device = device
        self.model_size = model_size
        self.sam_ckpt_dir = sam_ckpt_dir
        self._predictor = None

    def _load(self):
        if self._predictor is None:
            try:
                from sam2.sam2_image_predictor import SAM2ImagePredictor
            except ImportError as e:
                raise SegmentorUnavailable(
                    "sam2 not installed: pip install git+https://github.com/"
                    "facebookresearch/sam2.git") from e
            self._predictor = SAM2ImagePredictor(
                build_sam2_model(self.model_size, self.device, self.sam_ckpt_dir))
        return self._predictor

    def config(self) -> dict:
        # Resolve the checkpoint dir NOW rather than storing None: with
        # sam_ckpt_dir unset, build_sam2_model reads POPOE_SAM2_CKPT at load
        # time, so two runs under different env values would otherwise share a
        # cache key while loading different weights.
        from popoe.segmentor import default_ckpt_dir
        return {"model_size": self.model_size,
                "ckpt_dir": self.sam_ckpt_dir or default_ckpt_dir()}

    def masks_for_boxes(self, rgb: np.ndarray,
                        boxes: np.ndarray) -> list[tuple[np.ndarray, float]]:
        predictor = self._load()
        predictor.set_image(np.asarray(rgb, dtype=np.uint8))
        out = []
        for box in np.asarray(boxes, dtype=np.float64):
            masks, ious, _ = predictor.predict(box=box, multimask_output=True)
            bi = int(np.argmax(ious))
            out.append((masks[bi].astype(bool), float(ious[bi])))
        return out


class DinoV2ClsGemEmbedder:
    """DINOv2 class token + GeM-pooled foreground patch tokens.

    Both come out of ONE forward pass: MUSE needs the pair for every proposal
    and every template, and running the backbone twice would double the cost of
    the stage's dominant term.
    """

    def __init__(self, device: str = "cuda",
                 model_name: str = "dinov2_vitg14_reg",
                 grid: int = GRID, gem_p: float = DEFAULT_GEM_P,
                 fg_threshold: float = 0.4):
        self.device = device
        self.model_name = model_name
        self.grid = int(grid)
        self.gem_p = float(gem_p)
        # Mirrored on self (not only inside the scorer) so it shows up in the
        # component identity that keys cached output.
        self.fg_threshold = float(fg_threshold)
        self._patch_mask = PatchForegroundScorer(grid=grid, fg_threshold=fg_threshold)
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
                if _is_runtime_failure(e):
                    raise
                raise SegmentorUnavailable(
                    f"DINOv2 ({self.model_name}) load failed: {e}") from e
            self._mean = torch.tensor([0.485, 0.456, 0.406],
                                      device=self.device).view(1, 3, 1, 1)
            self._std = torch.tensor([0.229, 0.224, 0.225],
                                     device=self.device).view(1, 3, 1, 1)
        return self._model

    def config(self) -> dict:
        return {"model_name": self.model_name, "grid": self.grid,
                "gem_p": self.gem_p, "fg_threshold": self.fg_threshold}

    def embed(self, rgb_crop: np.ndarray,
              fg_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(cls (d,) L2-normed, gem (d,) raw)`` for one crop."""
        model = self.model      # guards torch: missing -> SegmentorUnavailable
        import torch

        x = torch.from_numpy(np.asarray(rgb_crop, dtype=np.uint8)).float()
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        x = (x - self._mean) / self._std
        with torch.no_grad():
            feats = model.forward_features(x)
        cls = torch.nn.functional.normalize(feats["x_norm_clstoken"][0], dim=0)
        tok = feats["x_norm_patchtokens"][0].detach().cpu().numpy()
        keep = self._patch_mask.foreground_patch_mask(np.asarray(fg_mask, dtype=bool))
        return cls.detach().cpu().numpy().astype(np.float64), \
            gem_pool(tok[keep], self.gem_p)


class MuseTemplateBank:
    """Per-class ``(cls, gem)`` bank over rendered template PNGs.

    Same PNG convention as CNOS-v3 (alpha channel when present, else a
    near-black background threshold), so one rendered template set serves both
    segmentors.
    """

    def __init__(self,
                 template_dir: str | Mapping[int, str] | Callable[[ObjectModel], str],
                 embedder: DinoV2ClsGemEmbedder):
        self.template_dir = template_dir
        self.embedder = embedder
        self._cache: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}

    def _dir_for(self, obj: ObjectModel) -> str:
        if callable(self.template_dir):
            return str(self.template_dir(obj))
        if isinstance(self.template_dir, MappingABC):
            return str(self.template_dir[obj.obj_id])
        return str(self.template_dir)

    def config(self) -> dict:
        dirs = self.template_dir
        return {"template_dir": (dirs if isinstance(dirs, (str, MappingABC))
                                 else repr(dirs)),
                "embedder": _component_identity(self.embedder)}

    def embeddings_for(self, obj: ObjectModel) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(cls_bank (T,d), gem_bank (T,d))`` for one class."""
        key = (obj.obj_id, self._dir_for(obj))
        if key in self._cache:
            return self._cache[key]
        try:
            from PIL import Image
        except ImportError as e:
            raise SegmentorUnavailable("Pillow is needed to load MUSE templates") from e

        cls_rows, gem_rows = [], []
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
            if crop is None:
                continue
            cls_v, gem_v = self.embedder.embed(crop, cmask)
            cls_rows.append(cls_v)
            gem_rows.append(gem_v)
        if not cls_rows:
            raise SegmentorUnavailable(f"no usable MUSE templates in {key[1]}")
        bank = (np.stack(cls_rows), np.stack(gem_rows))
        self._cache[key] = bank
        return bank


# ── The live segmentor ──────────────────────────────────────────────────

@dataclass
class MuseSceneResult:
    """Everything MUSE computes for ONE frame, shared by all its classes."""
    masks: list[np.ndarray]
    p_obj: np.ndarray            # (P,)
    extents: np.ndarray          # (P,) metres
    s_abs: np.ndarray            # (P, C)
    s_rel: np.ndarray            # (P, C)
    s_joint: np.ndarray          # (P, C)
    s_final: np.ndarray          # (P, C)
    obj_ids: tuple[int, ...]     # column order

    @property
    def n_proposals(self) -> int:
        return len(self.masks)


class MuseSegmentor:
    """MUSE reimplementation as a live ``Segmentor``.

    Two things make this NOT a copy of ``CNOSv3Segmentor``'s shape, both
    consequences of MUSE scoring classes jointly:

    1. **The relative score needs every class at once.** ``Segmentor.segment``
       is a per-object contract, so classes are registered up front and
       ``segment`` serves one column of a matrix computed for all of them. With
       a single registered class ``S_rel`` collapses to the constant 1 and the
       method silently becomes "S_abs with extra steps" — so that configuration
       must be asked for explicitly (`allow_single_class`).
    2. **Proposals are per-frame, not per-object.** Grounding DINO and SAM2
       would otherwise re-run for every object in the same image. Results are
       memoised per frame, keyed by CONTENT (the cache.py invariant: never by
       list position or scene_id, which real captures often leave at -1).

    `score` is ``S_final``; ``Detection.descriptor`` carries the breakdown
    (`DESCRIPTOR_FIELDS`). As with every segmentor, that score is comparable
    only within MUSE — see ``interfaces.Detection``.
    """

    source = MUSE_SOURCE

    def __init__(self, classes: Sequence[MuseClass],
                 proposer: Optional[GroundingDinoBoxProposer] = None,
                 refiner: Optional[SAM2BoxMaskRefiner] = None,
                 embedder: Optional[DinoV2ClsGemEmbedder] = None,
                 template_bank: Optional[MuseTemplateBank] = None,
                 size_gate: Optional[DepthSizeGate] = None,
                 alpha: float = DEFAULT_ALPHA,
                 beta: float = DEFAULT_BETA,
                 tau: float = DEFAULT_TAU,
                 gamma: float = DEFAULT_GAMMA,
                 n_masks: int = 2,
                 allow_single_class: bool = False,
                 cache_frames: int = 2):
        classes = list(classes)
        if not classes:
            raise ValueError("MuseSegmentor needs at least one registered class")
        if len(classes) < 2 and not allow_single_class:
            raise ValueError(
                "MUSE's relative score is a softmax ACROSS classes; with one "
                "registered class it is the constant 1 and S_joint degenerates "
                "to beta*S_abs. Register the other candidate classes, or pass "
                "allow_single_class=True to choose that degenerate variant "
                "deliberately.")
        ids = [c.obj.obj_id for c in classes]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate obj_id in registered classes: {ids}")
        self.classes = classes
        self._index = {c.obj.obj_id: i for i, c in enumerate(classes)}
        self.proposer = proposer
        self.refiner = refiner
        self.embedder = embedder
        self.template_bank = template_bank
        self.size_gate = size_gate or DepthSizeGate(min_pixels=DEFAULT_MIN_PIXELS)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.n_masks = _check_n_masks(n_masks)
        self.cache_frames = int(cache_frames)
        self._frames: dict[str, MuseSceneResult] = {}

    # -- config identity (belongs in any cache key that stores our output) --
    def config(self) -> dict:
        """Everything that changes this segmentor's OUTPUT for a fixed frame.

        Completeness is the whole point: this is both the per-frame memo key and
        what a `popoe.cache` user keys stored MUSE output on, so a knob missing
        here aliases two genuinely different configurations (ARCHITECTURE.md —
        "anything that selects a method belongs in the cache key"). Hence the
        class DIAMETERS (they drive the size gate, not just the class list),
        every gate field, and each component's public settings.

        Known limit: template directories are keyed by PATH, not by content —
        editing the PNGs under a path in place will not invalidate. Point a new
        directory at edited templates.
        """
        return {
            "source": self.source,
            "alpha": self.alpha, "beta": self.beta,
            "tau": self.tau, "gamma": self.gamma,
            "n_masks": self.n_masks,
            "classes": {c.obj.obj_id: (c.template_dir, float(c.obj.diameter))
                        for c in self.classes},
            # ORDERED, because the dict above is not: registration order fixes
            # the column order of MuseSceneResult.s_* / obj_ids, so [9, 14] and
            # [14, 9] must not share a key — an external cache serving one
            # matrix for the other would swap every class's scores.
            "class_order": [c.obj.obj_id for c in self.classes],
            "size_gate": _component_identity(self.size_gate),
            "proposer": _component_identity(self.proposer),
            "refiner": _component_identity(self.refiner),
            "embedder": _component_identity(self.embedder),
            "template_bank": _component_identity(self.template_bank),
        }

    # -- per-frame computation ------------------------------------------
    def _gate_union(self, scene: Scene,
                    mask: np.ndarray) -> tuple[bool, Optional[float]]:
        """Keep a proposal plausible for AT LEAST ONE registered class.

        The extent is computed once and tested against each class's interval —
        calling ``DepthSizeGate.accepts`` per class would recompute the same
        unprojection C times. This is the true union, not
        ``[min(lo_c), max(hi_c)]``: those differ once the registered diameters
        span more than 4.4x (the ratio band's width), where the hull admits
        proposals plausible for NO class.
        """
        gate = self.size_gate
        if scene.depth is None or scene.K is None:
            return False, None
        if int(mask.sum()) < gate.min_pixels:
            return False, None
        extent = gate.extent_3d(mask, scene.depth, scene.K)
        if extent is None:
            return False, None
        for c in self.classes:
            d = float(c.obj.diameter)
            if gate.min_extent_ratio * d <= extent <= gate.max_extent_ratio * d:
                return True, extent
        return False, extent

    def _require(self, component, what: str):
        if component is None:
            raise SegmentorUnavailable(f"MUSE needs {what}")
        return component

    def scene_result(self, scene: Scene) -> MuseSceneResult:
        """Run (or reuse) the whole MUSE stage for one frame, all classes."""
        key = fingerprint(scene.rgb, scene.depth, scene.K, self.config())
        cached = self._frames.get(key)
        if cached is not None:
            return cached

        proposer = self._require(self.proposer, "a Grounding DINO box proposer")
        refiner = self._require(self.refiner, "a SAM2 box-prompt mask refiner")
        embedder = self._require(self.embedder, "a DINOv2 cls+patch embedder")
        bank = self._require(self.template_bank, "a template bank")

        boxes, objectness = proposer.propose(scene)
        masks, p_obj, extents = [], [], []
        if len(boxes):
            for (mask, _sam_iou), p in zip(refiner.masks_for_boxes(scene.rgb, boxes),
                                           objectness):
                ok, extent = self._gate_union(scene, mask)
                if not ok:
                    continue
                masks.append(mask)
                p_obj.append(float(p))
                extents.append(float(extent))

        # Masks are shared by every class served from this frame, so a caller
        # writing into one would corrupt another object's detection. Nothing in
        # popoe writes into a Detection.mask; freezing makes a future one loud.
        for mask in masks:
            mask.flags.writeable = False

        banks = ([bank.embeddings_for(c.obj) for c in self.classes]
                 if masks else [])          # no proposals: don't load templates
        s_abs = np.full((len(masks), len(self.classes)), -np.inf)
        for pi, mask in enumerate(masks):
            crop, crop_mask = square_crop(scene.rgb, mask)
            if crop is None:
                continue                       # row stays -inf: never a winner
            cls_q, gem_q = embedder.embed(crop, crop_mask)
            for ci, (cls_bank, gem_bank) in enumerate(banks):
                s_abs[pi, ci] = absolute_score(cls_q, gem_q, cls_bank, gem_bank,
                                               self.alpha)

        if len(masks):
            s_rel = relative_scores(s_abs, self.tau)
            s_joint = joint_scores(s_abs, s_rel, self.beta)
            s_final = final_scores(s_joint, np.asarray(p_obj), self.gamma)
        else:
            s_rel = s_joint = s_final = np.zeros((0, len(self.classes)))

        result = MuseSceneResult(
            masks=masks,
            p_obj=np.asarray(p_obj, dtype=np.float64),
            extents=np.asarray(extents, dtype=np.float64),
            s_abs=s_abs, s_rel=s_rel, s_joint=s_joint, s_final=s_final,
            obj_ids=tuple(c.obj.obj_id for c in self.classes),
        )
        if self.cache_frames > 0:
            while len(self._frames) >= self.cache_frames:
                self._frames.pop(next(iter(self._frames)))
            self._frames[key] = result
        return result

    def detections_for(self, scene: Scene, obj_id: int,
                       n_masks: Optional[int] = None) -> list[Detection]:
        try:
            ci = self._index[int(obj_id)]
        except KeyError:
            raise ValueError(
                f"obj_id {obj_id} is not registered with this MuseSegmentor "
                f"(have {sorted(self._index)}). MUSE scores classes jointly, so "
                f"classes cannot be added per call.") from None
        result = self.scene_result(scene)
        k = self.n_masks if n_masks is None else _check_n_masks(n_masks)
        order = np.argsort(-result.s_final[:, ci])[:k] if result.n_proposals else []
        dets = []
        for pi in order:
            if not np.isfinite(result.s_abs[pi, ci]):
                continue
            mask = result.masks[pi]
            dets.append(Detection(
                mask=mask,
                score=float(result.s_final[pi, ci]),
                bbox=_bbox_xyxy(mask),
                source=self.source,
                descriptor=np.asarray([result.s_abs[pi, ci], result.s_rel[pi, ci],
                                       result.p_obj[pi], result.extents[pi]],
                                      dtype=np.float32),
            ))
        return dets

    def segment(self, scene: Scene, obj: ObjectModel) -> list[Detection]:
        return self.detections_for(scene, obj.obj_id)


def build_muse_segmentor(classes: Sequence[MuseClass],
                         *,
                         device: str = "cuda",
                         gdino_model: str = DEFAULT_GDINO_MODEL,
                         prompt: str = DEFAULT_PROMPT,
                         box_threshold: float = 0.15,
                         text_threshold: float = 0.15,
                         sam2_size: str = "large",
                         sam_ckpt_dir: Optional[str] = None,
                         dinov2_model: str = "dinov2_vitg14_reg",
                         min_pixels: int = DEFAULT_MIN_PIXELS,
                         **kwargs) -> MuseSegmentor:
    """Wire the real GPU components. Construction stays lazy — nothing loads
    until the first ``segment``, so an unavailable backend surfaces at the call
    that needed it, not at import."""
    embedder = DinoV2ClsGemEmbedder(device=device, model_name=dinov2_model)
    return MuseSegmentor(
        classes,
        proposer=GroundingDinoBoxProposer(gdino_model, prompt, box_threshold,
                                          text_threshold, device),
        refiner=SAM2BoxMaskRefiner(device, sam2_size, sam_ckpt_dir),
        embedder=embedder,
        template_bank=MuseTemplateBank({c.obj.obj_id: c.template_dir
                                        for c in classes}, embedder),
        size_gate=DepthSizeGate(min_pixels=min_pixels),
        **kwargs,
    )


# ── The producer half: dump to a BOP detections JSON ────────────────────

def encode_rle(mask: np.ndarray) -> dict:
    """Compressed COCO RLE, JSON-safe (`counts` as str, not bytes)."""
    try:
        from pycocotools import mask as coco_mask
    except ImportError as e:
        raise SegmentorUnavailable(
            "pycocotools is needed to write detections JSON") from e
    rle = coco_mask.encode(np.asfortranarray(np.asarray(mask, dtype=np.uint8)))
    counts = rle["counts"]
    return {"size": [int(s) for s in rle["size"]],
            "counts": counts.decode("ascii") if isinstance(counts, bytes) else counts}


def muse_records(scene: Scene, segmentor: MuseSegmentor, *,
                 n_masks: Optional[int] = None,
                 source: str = MUSE_SOURCE) -> list[dict]:
    """BOP-format detection records for every registered class of one frame.

    This is the bridge between MUSE's two halves: the live segmentor's masks
    become the same artefact CNOS / SAM-6D / NIDS publish, so downstream code
    (union segmentor, mask-AP evaluation) treats MUSE like any other source.
    """
    _check_source(source)
    out = []
    for c in segmentor.classes:
        for det in segmentor.detections_for(scene, c.obj.obj_id, n_masks=n_masks):
            rec = {
                "scene_id": int(scene.scene_id),
                "image_id": int(scene.im_id),
                "category_id": int(c.obj.obj_id),
                "score": float(det.score),
                "segmentation": encode_rle(det.mask),
                "source": source,
            }
            if det.bbox is not None:
                x0, y0, x1, y1 = det.bbox
                rec["bbox"] = [float(x0), float(y0),
                               float(x1 - x0), float(y1 - y0)]  # COCO xywh
            if det.descriptor is not None:
                rec["muse"] = {k: float(v) for k, v
                               in zip(DESCRIPTOR_FIELDS, det.descriptor)}
            out.append(rec)
    return out


def write_muse_detections(records: Sequence[dict], output_json: str) -> str:
    """Write detection records to disk, refusing any that claim the official name.

    The writer re-checks rather than trusting `muse_records`: it is the last gate
    before an artefact exists on disk, and records can be built or edited by a
    caller that never went through the stamping helpers.
    """
    for i, rec in enumerate(records):
        if "source" in rec:
            try:
                _check_source(rec["source"])
            except ValueError as e:
                raise ValueError(f"detection record {i}: {e}") from None
    out_dir = os.path.dirname(os.path.abspath(output_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(list(records), f)
    return output_json


class MuseDetectionsSegmentor(BOPDetectionsSegmentor):
    """File-backed Segmentor over a dumped MUSE detections JSON.

    The consumption end of the producer half: once ``write_muse_detections``
    has run, MUSE is an ordinary named source and needs no GPU, no Grounding
    DINO and no templates to replay — exactly like ``cnos`` / ``sam6d`` /
    ``nids``. Prefer this in evaluation and in multi-source unions, where the
    live segmentor would otherwise re-run the whole cascade per frame.
    """

    source = MUSE_SOURCE

    def __init__(self, detections_json: str, topk: int = 2,
                 merge_labels: Optional[dict] = None,
                 iou_dedupe: float = 0.9,
                 min_pixels: int = 100,
                 source: str = MUSE_SOURCE):
        super().__init__(
            detections_json, topk=topk, merge_labels=merge_labels,
            iou_dedupe=iou_dedupe, min_pixels=min_pixels,
            source=_check_source(source))


# ── CLI ─────────────────────────────────────────────────────────────────

def _parse_classes(spec: str) -> list[tuple[int, str]]:
    """'9=/tpl/obj_000009,14=/tpl/obj_000014' -> [(9, dir), (14, dir)]."""
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"--classes entry needs 'obj_id=template_dir', got {item!r}")
        oid, path = item.split("=", 1)
        out.append((int(oid), path.strip()))
    return out


def _classes_from_args(pairs: Sequence[tuple[int, str]], models_info: str,
                       models_dir: Optional[str]) -> list[MuseClass]:
    with open(models_info) as f:
        minfo = json.load(f)
    classes = []
    for oid, tpl in pairs:
        if str(oid) not in minfo:
            raise SystemExit(f"obj_id {oid} not in {models_info}")
        mesh = (os.path.join(models_dir, f"obj_{oid:06d}.ply")
                if models_dir else "")
        classes.append(MuseClass(
            obj=ObjectModel(obj_id=oid, mesh_path=mesh,
                            diameter=float(minfo[str(oid)]["diameter"]) / 1000.0),
            template_dir=tpl))
    return classes


def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run the MUSE reimplementation on one frame and write a "
                    "BOP-format detections JSON (source='muse-repro').")
    ap.add_argument("--frame", required=True,
                    help="frame manifest JSON (rgb_path, depth_path, K, depth_scale)")
    ap.add_argument("--classes", required=True,
                    help="'obj_id=template_dir' pairs, comma separated (>=2 "
                         "so the relative score is meaningful)")
    ap.add_argument("--models-info", required=True,
                    help="BOP models_info.json — diameters for the size gate")
    ap.add_argument("--models-dir", default=None,
                    help="BOP models dir, if the records should carry mesh paths")
    ap.add_argument("--out", required=True, help="output detections JSON")
    ap.add_argument("--topk", type=int, default=2, help="masks kept per class")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gdino", default=DEFAULT_GDINO_MODEL)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--box-threshold", type=float, default=0.15)
    ap.add_argument("--text-threshold", type=float, default=0.15)
    ap.add_argument("--sam2-size", default="large",
                    help="tiny|small|base|large (checkpoint dir: POPOE_SAM2_CKPT)")
    ap.add_argument("--sam-ckpt-dir", default=None)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--beta", type=float, default=DEFAULT_BETA)
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU)
    ap.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    ap.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    ap.add_argument("--allow-single-class", action="store_true")
    args = ap.parse_args(argv)

    from popoe.datasets.frames import load_scene_from_manifest
    from popoe.segmentor_detections import load_bop_detections

    scene = load_scene_from_manifest(args.frame)
    classes = _classes_from_args(_parse_classes(args.classes),
                                 args.models_info, args.models_dir)
    seg = build_muse_segmentor(
        classes, device=args.device, gdino_model=args.gdino, prompt=args.prompt,
        box_threshold=args.box_threshold, text_threshold=args.text_threshold,
        sam2_size=args.sam2_size, sam_ckpt_dir=args.sam_ckpt_dir,
        min_pixels=args.min_pixels,
        alpha=args.alpha, beta=args.beta, tau=args.tau, gamma=args.gamma,
        n_masks=args.topk, allow_single_class=args.allow_single_class)

    records = muse_records(scene, seg, n_masks=args.topk)
    result = seg.scene_result(scene)
    print(f"proposals kept after size gate: {result.n_proposals}")
    for ci, c in enumerate(seg.classes):
        if result.n_proposals:
            best = int(np.argmax(result.s_final[:, ci]))
            print(f"  obj {c.obj.obj_id}: best s_final={result.s_final[best, ci]:.4f} "
                  f"s_abs={result.s_abs[best, ci]:.3f} "
                  f"s_rel={result.s_rel[best, ci]:.3f} "
                  f"P(O)={result.p_obj[best]:.3f} "
                  f"extent={result.extents[best] * 1000:.0f}mm")
        else:
            print(f"  obj {c.obj.obj_id}: no proposals survived the size gate")
    write_muse_detections(records, args.out)
    # Load once through the production loader so a schema error surfaces here.
    load_bop_detections(args.out, source=MUSE_SOURCE)
    print(f"wrote {len(records)} detections -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "DESCRIPTOR_FIELDS",
    "DinoV2ClsGemEmbedder",
    "GroundingDinoBoxProposer",
    "MUSE_SOURCE",
    "MuseClass",
    "MuseDetectionsSegmentor",
    "MuseSceneResult",
    "MuseSegmentor",
    "MuseTemplateBank",
    "SAM2BoxMaskRefiner",
    "absolute_score",
    "build_muse_segmentor",
    "encode_rle",
    "final_scores",
    "gem_pool",
    "joint_scores",
    "muse_records",
    "relative_scores",
    "tanimoto",
    "write_muse_detections",
]
