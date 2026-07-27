"""BOP official-detections Segmentor — with confusable-pair label pooling.

Reads a BOP-format detections JSON (e.g. the public CNOS-FastSAM files) and
serves per-target candidate masks, reproducing the evaluated best practice:

  * top-K per (source, label) bucket, sorted by detector score;
  * near-duplicate masks dropped (IoU > `iou_dedupe`);
  * **label pooling** for confusable same-shape/different-size pairs
    (`merge_labels={19: [19, 20], 20: [19, 20]}` for the YCB-V clamp pair):
    the detector matches by appearance and cannot tell such pairs apart, so an
    object's true instance frequently sits under its partner's label (measured
    73-86% of the time for YCB-V obj20). Pooling both labels' top-K recovers
    it; a size-aware scorer (popoe.scoring.ChampionScorer) arbitrates.

Pure numpy + pycocotools; no GPU.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from popoe.interfaces import Detection, ObjectModel, Scene


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return (np.logical_and(a, b).sum() / union) if union else 0.0


# ── Loading: field coercion + RLE compatibility ──────────────────────────
#
# BOP detection files in the wild are NOT uniformly typed. The public
# CNOS-FastSAM files are already numeric with list-valued (uncompressed) RLE;
# the NIDS-Net WA_Sappe release hosted on Box is documented to arrive with
# EVERY field stringified ("scene_id": "48", "score": "0.74...", bbox as a
# stringified list). Both must load without special-casing at the call site.
#
# Why coerce at load time rather than trust the values: an un-coerced string
# category poisons the *silent* path, not the loud one. `d["category_id"] in
# labels` with category_id=="1" and labels==[1] is simply False -> the image
# yields zero candidates and looks like "object not in frame", never an error.
# (`-d["score"]` on a str would at least raise.) Coercion turns a silent miss
# into correct matching. See test_detections_load.py.

def _to_int(x) -> int:
    """Coerce ids/counts to int, accepting int, float, or numeric string
    ('48' and '48.0' both -> 48). A NON-integral value ('1.9', '1e-1') RAISES
    rather than truncating: a truncated scene_id/image_id/category_id silently
    routes to the wrong image/object, the exact silent miss this loader exists
    to prevent."""
    f = float(x)
    if not f.is_integer():
        raise ValueError(f"expected an integer-valued field, got {x!r}")
    return int(f)


def _to_float(x) -> float:
    return float(x)


def _coerce_bbox(b):
    """bbox may be a real list [x, y, w, h] or a stringified one "[x, y, w, h]"."""
    if isinstance(b, str):
        b = json.loads(b)
    return [float(v) for v in b]


def _bbox_xywh_to_xyxy(b):
    if b is None:
        return None
    if len(b) != 4:
        raise ValueError(f"bbox must be [x, y, w, h], got {b!r}")
    x, y, w, h = [float(v) for v in b]
    return (x, y, x + w, y + h)


def _normalize_segmentation(seg):
    """Return an RLE dict with a numeric `size` and `counts` that is either a
    list of ints (uncompressed) or a str/bytes (compressed COCO RLE).

    Handles the stringified variants of the WA_Sappe release:
      * `size` as "[480, 640]"  -> [480, 640]
      * `counts` as "[6628, 2, ...]" (a stringified uncompressed run list)
        -> parsed back to a list.
    A genuine compressed-RLE string is passed through as-is — including ones
    that START WITH '[': COCO's compressed counts are LEB128 bytes mapped into
    printable ASCII, so e.g. "[1d0Q6" is a valid compressed RLE, NOT a JSON
    list. The discriminator is therefore "parses as a JSON array of numbers",
    tried and rolled back on failure, not a cheap first-character check.
    """
    if not isinstance(seg, dict):
        return seg
    size = seg.get("size")
    if isinstance(size, str):
        size = json.loads(size)
    size = [int(s) for s in size]
    counts = seg.get("counts")
    if isinstance(counts, str) and counts.lstrip().startswith("["):
        try:
            parsed = json.loads(counts)
        except json.JSONDecodeError:
            parsed = None                    # compressed RLE that begins with '['
        if isinstance(parsed, list):
            counts = parsed                  # stringified uncompressed run list
    if isinstance(counts, list):
        counts = [int(c) for c in counts]    # stringified ints -> ints
    return {"counts": counts, "size": size}


def _normalize_mask_path(path, base_dir: str | None):
    p = os.fspath(path)
    if base_dir is not None and not os.path.isabs(p):
        p = os.path.normpath(os.path.join(base_dir, p))
    return {"format": "path", "path": p}


def _normalize_detection_mask(d: dict, base_dir: str | None):
    """Normalise the mask field from either BOP or real-scene schema.

    BOP files use `segmentation` for COCO RLE. Real-scene files may use either
    the same field or `mask`:
      * {"format": "rle", "size": [H, W], "counts": ...}
      * {"size": [H, W], "counts": ...}
      * "relative/or/absolute/mask.png" / "mask.npy"
      * {"format": "path", "path": "mask.png"}
    """
    if "segmentation" in d:
        return _normalize_segmentation(d["segmentation"])
    if "mask" in d:
        mask = d["mask"]
        if isinstance(mask, str):
            return _normalize_mask_path(mask, base_dir)
        if isinstance(mask, dict):
            if "counts" in mask and "size" in mask:
                return _normalize_segmentation(mask)
            if mask.get("format") == "rle":
                payload = mask.get("rle", mask)
                return _normalize_segmentation(payload)
            if "path" in mask:
                return _normalize_mask_path(mask["path"], base_dir)
        raise ValueError(f"unsupported detection mask schema: {mask!r}")
    if "mask_path" in d:
        return _normalize_mask_path(d["mask_path"], base_dir)
    raise KeyError("detection record needs `segmentation`, `mask`, or `mask_path`")


def _records_from_payload(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("detections", "annotations", "instances", "results"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise TypeError(
        "detections JSON must be a list, or a dict with one of: "
        "detections, annotations, instances, results"
    )


def load_bop_detections(path: str, source: str | None = None) -> list[dict]:
    """Load a detections JSON into normalised, numerically-typed
    records, robust to the fully-stringified WA_Sappe (NIDS) variant.

    The primary input is BOP format
    `{scene_id, image_id, category_id, score, segmentation}`. For real RGB-D
    captures, the same loader also accepts `mask` / `mask_path` as a 2D mask
    field. In all cases, each returned record has int scene_id/image_id/
    category_id, float score, and a normalised `segmentation` entry that
    `decode_detection_mask` can turn into a bool mask. `bbox` (if present) is
    coerced to a float list. Any OTHER fields on the
    record (e.g. BOP's per-detection `time`) are carried through untouched, so
    the loader can also normalise-and-rewrite without dropping metadata.
    `source` is set from the argument when given (it stamps the origin for
    multi-source union), else from the record's own "source" field when
    present, else left absent (the segmentor buckets those under "_")."""
    base_dir = os.path.dirname(os.path.abspath(path))
    with open(path) as f:
        raw = json.load(f)
    records = []
    for d in _records_from_payload(raw):
        rec = dict(d)                        # keep unknown fields (time, ...)
        rec["scene_id"] = _to_int(d["scene_id"])
        rec["image_id"] = _to_int(d["image_id"])
        rec["category_id"] = _to_int(d["category_id"])
        rec["score"] = _to_float(d["score"])
        rec["segmentation"] = _normalize_detection_mask(d, base_dir)
        if "bbox" in d:
            rec["bbox"] = _coerce_bbox(d["bbox"])
        if source is not None:
            rec["source"] = source
        records.append(rec)
    return records


def _read_mask_image(path: str) -> np.ndarray:
    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    else:
        try:
            import cv2
        except ImportError:
            cv2 = None
        if cv2 is not None:
            arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise FileNotFoundError(path)
        else:
            try:
                from PIL import Image
            except ImportError as e:
                raise ImportError(
                    f"reading mask image files needs opencv-python or Pillow; "
                    f"cannot load {path!r}"
                ) from e
            arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = np.any(arr > 0, axis=2)
    return arr.astype(bool)


def decode_detection_mask(seg: dict) -> np.ndarray:
    """Decode a BOP RLE `segmentation` dict to a (H, W) bool mask.

    Accepts BOTH RLE encodings that appear in BOP files:
      * uncompressed — `counts` is a list of run lengths (column-major, COCO
        convention). pycocotools decodes these via `frPyObjects(seg, h, w)`,
        which is byte-identical to a manual column-major run decode (verified).
      * compressed — `counts` is a str/bytes COCO RLE, decoded directly.

    The dict-vs-string distinction is exactly the "counts gotcha": a list
    `counts` is an uncompressed RLE and MUST go through `frPyObjects` (passing
    it to `decode` treats it as already-compressed and returns garbage)."""
    if seg.get("format") == "path":
        return _read_mask_image(seg["path"])
    from pycocotools import mask as coco_mask
    h, w = int(seg["size"][0]), int(seg["size"][1])
    counts = seg["counts"]
    rle = coco_mask.frPyObjects(seg, h, w) if isinstance(counts, list) else seg
    m = coco_mask.decode(rle).astype(bool)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m


# ── Pluggable file-based detection backends ──────────────────────────────
#
# CNOS-FastSAM, SAM-6D ISM and NIDS-Net all publish the SAME artefact: a
# detections JSON. They are not different *code* paths here, only different
# files under different NAMES. So a "segmentation backend" at this layer is just
# a named file source; the actual producer may live in its own environment
# (NIDS-Net's GroundingDINO/SAM/DINOv2 stack, a CNOS-style script, a manual
# SAM-prompt tool). Naming each source keeps provenance on every mask
# (`Detection.source`), exactly as the fallback chain does
# (segmentor.FirstAvailableSegmentor) — see ARCHITECTURE.md.

@dataclass(frozen=True)
class DetectionSource:
    """One named BOP-detections file. `name` is the provenance tag stamped onto
    every `Detection` it yields (canonically 'cnos' / 'sam6d' / 'nids'); `path`
    is the JSON. It is the config-level handle a caller selects BY NAME and
    composes with others (see BOPDetectionsSegmentor's `sources=`)."""
    name: str
    path: str


def _coerce_sources(sources) -> list[DetectionSource]:
    """Accept the ways a config can name detection sources and normalise to a
    list[DetectionSource]:
      * dict            {"nids": path, "cnos": path}
      * (name, path) tuples / DetectionSource instances, in a list
      * "name=path" strings (CLI-friendly), in a list
    """
    out: list[DetectionSource] = []
    items = sources.items() if isinstance(sources, dict) else sources
    for s in items:
        if isinstance(sources, dict):
            out.append(DetectionSource(s[0], s[1]))
        elif isinstance(s, DetectionSource):
            out.append(s)
        elif isinstance(s, str) and "=" in s:
            name, path = s.split("=", 1)
            out.append(DetectionSource(name, path))
        elif isinstance(s, (tuple, list)) and len(s) == 2:
            out.append(DetectionSource(str(s[0]), str(s[1])))
        else:
            raise TypeError(
                f"cannot read detection source spec {s!r}; expected a "
                f"DetectionSource, (name, path), or 'name=path'")
    if not out:
        raise ValueError("no detection sources given")
    names = [s.name for s in out]
    if any(not n for n in names):
        raise ValueError("every detection source needs a non-empty name")
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate source names would collide: {names}")
    return out


class BOPDetectionsSegmentor:
    """Segmentor over one OR several BOP default-detections JSON files.

    Args:
        detections_json: path to a single BOP-format detections file (list of
            dicts with scene_id / image_id / category_id / score / segmentation
            RLE). Back-compatible single-source form. Mutually exclusive with
            `sources`.
        topk: detections kept per (source, label) bucket.
        merge_labels: {obj_id: [obj_id, partner_id, ...]} — pool these labels'
            candidates for the given object. Default None (no pooling).
        iou_dedupe: drop a mask whose IoU with an already-kept one exceeds this.
        min_pixels: drop masks smaller than this (unreliable geometry).
        sources: a config of NAMED backends to union — dict {name: path},
            DetectionSource / (name, path) list, or "name=path" strings (see
            `_coerce_sources`). Each record is tagged with its source name, so
            the per-(source, label) bucketing keeps top-K PER SOURCE and every
            returned `Detection.source` says which backend produced it.
        source: provenance tag for the single-file form (default: the class
            tag 'bop-detections', which preserves historical `Detection.source`).
        size_select: optional mask-stage size arbitration for confusable
            same-shape pairs (YCB-V clamps). ``None`` (default) keeps detector
            scores unchanged — formal BOP headline path. ``"soft"`` multiplies
            appearance by competitive diameter affinity; ``"nearest"`` keeps
            only masks whose nearest CAD in ``confusable_diameters`` is the
            query (with appearance fallback if none match, unless
            ``size_select_fallback=False``). Needs ``scene.depth`` + ``scene.K``.
            Complementary to registration-stage ``ChampionScorer(size_aware)``.
        confusable_diameters: ``{obj_id: diameter_m}`` for the confusable set
            (e.g. ``{19: 0.175, 20: 0.217}``). Required when ``size_select`` set.
        size_select_fallback: for ``nearest``, fall back to pure appearance
            ranking when no mask is assigned to the query (default True).
    """

    source = "bop-detections"

    def __init__(self, detections_json: str | None = None, topk: int = 2,
                 merge_labels: dict | None = None, iou_dedupe: float = 0.9,
                 min_pixels: int = 100, *, sources=None,
                 source: str | None = None,
                 size_select: str | None = None,
                 confusable_diameters: dict | None = None,
                 size_select_fallback: bool = True):
        if (detections_json is None) == (sources is None):
            raise ValueError("pass exactly one of detections_json or sources")
        if source is not None and not source:
            raise ValueError("source name must be non-empty")
        if size_select not in (None, "soft", "nearest"):
            raise ValueError(
                f"size_select must be None/'soft'/'nearest', got {size_select!r}"
            )
        if size_select is not None and not confusable_diameters:
            raise ValueError("size_select requires confusable_diameters")
        self.topk = topk
        self.merge_labels = merge_labels or {}
        self.iou_dedupe = iou_dedupe
        self.min_pixels = min_pixels
        self.size_select = size_select
        self.confusable_diameters = (
            {int(k): float(v) for k, v in confusable_diameters.items()}
            if confusable_diameters is not None else None
        )
        self.size_select_fallback = bool(size_select_fallback)
        self._by_img: dict = {}
        if sources is not None:
            self.sources = _coerce_sources(sources)
        else:
            # Single file: one source whose name is the given tag, or the class
            # default 'bop-detections' — which every record is stamped with
            # (overwriting any stray per-record "source"), so the single-file
            # form keeps its historical, uniform Detection.source.
            self.sources = [DetectionSource(source or self.source,
                                            detections_json)]
        for s in self.sources:
            for d in load_bop_detections(s.path, source=s.name):
                self._by_img.setdefault(
                    (d["scene_id"], d["image_id"]), []).append(d)

    def segment(self, scene: Scene, obj: ObjectModel) -> list[Detection]:
        labels = self.merge_labels.get(obj.obj_id, [obj.obj_id])
        cands = [d for d in self._by_img.get((scene.scene_id, scene.im_id), [])
                 if d["category_id"] in labels]
        if not cands:
            return []
        buckets: dict = {}
        for d in cands:
            buckets.setdefault((d.get("source", "_"), d["category_id"]), []).append(d)
        picked = []
        for lst in buckets.values():
            picked.extend(sorted(lst, key=lambda d: -d["score"])[: self.topk])

        # The N-way top-M union does NOT filter ACROSS sources (FreeZe's
        # "top-M union without filtering"): two sources may propose the same
        # region and BOTH are kept, so the feature-aware scorer disposes with
        # every source's evidence intact. iou_dedupe is therefore scoped
        # PER SOURCE — a single backend still drops its own near-duplicates,
        # which for the one-source form is byte-identical to before.
        dets: list[Detection] = []
        kept_by_source: dict = {}
        for d in sorted(picked, key=lambda d: -d["score"]):
            m = decode_detection_mask(d["segmentation"])
            if m.sum() < self.min_pixels:
                continue
            src = d.get("source", self.source)
            kept = kept_by_source.setdefault(src, [])
            if any(_mask_iou(m, prev) > self.iou_dedupe for prev in kept):
                continue
            kept.append(m)
            dets.append(Detection(mask=m, score=float(d["score"]),
                                  bbox=_bbox_xywh_to_xyxy(d.get("bbox")),
                                  source=src))
        if self.size_select is not None:
            dets = self._apply_size_select(scene, obj, dets)
        return dets

    def _apply_size_select(
        self, scene: Scene, obj: ObjectModel, dets: list[Detection]
    ) -> list[Detection]:
        """Re-score / filter masks by measured 3D extent vs confusable diameters."""
        if not dets:
            return dets
        if scene.depth is None or scene.K is None:
            return dets
        if self.confusable_diameters is None:
            return dets
        if int(obj.obj_id) not in self.confusable_diameters:
            # Object not in the confusable set — leave scores alone.
            return dets

        from popoe.segmentor_cnos_lab import (
            DepthSizeGate,
            DiameterSizeModel,
            select_by_nearest_diameter,
            select_by_soft_affinity,
        )

        gate = DepthSizeGate(min_pixels=1, min_points=50)
        model = DiameterSizeModel()
        diams = self.confusable_diameters
        apps = [float(d.score) for d in dets]
        exts: list = []
        for d in dets:
            if int(d.mask.sum()) < self.min_pixels:
                exts.append(None)
            else:
                exts.append(gate.extent_3d(d.mask, scene.depth, scene.K))

        if self.size_select == "soft":
            rivals = [diams[k] for k in diams if k != int(obj.obj_id)]
            # Re-score every candidate; keep all, sorted by size-aware score.
            rescored = []
            for d, app, ext in zip(dets, apps, exts):
                if ext is None:
                    s = 0.0
                else:
                    aff_q = model.affinity(ext, diams[int(obj.obj_id)])
                    aff_sum = aff_q + sum(model.affinity(ext, r) for r in rivals)
                    w = aff_q / aff_sum if aff_sum > 0 else 0.0
                    s = app * w
                rescored.append(Detection(
                    mask=d.mask, score=float(s), bbox=d.bbox, source=d.source,
                    descriptor=d.descriptor,
                ))
            rescored.sort(key=lambda x: -x.score)
            return rescored

        # nearest
        pick = select_by_nearest_diameter(
            apps, exts, int(obj.obj_id), diams, model,
            fallback_appearance=self.size_select_fallback,
        )
        if pick is None:
            return []
        # Promote the selected mask to the front with its (appearance) score;
        # demote others so multi-hyp pipelines still see them but ranked lower.
        ordered = []
        chosen = dets[pick.index]
        ordered.append(Detection(
            mask=chosen.mask, score=float(chosen.score),
            bbox=chosen.bbox, source=chosen.source,
            descriptor=chosen.descriptor,
        ))
        for i, d in enumerate(dets):
            if i == pick.index:
                continue
            # Keep non-winners only when fallback path retained the full pool
            # for multi-hypothesis pose; mark them with a tiny score so a
            # single-mask consumer takes the nearest winner.
            ordered.append(Detection(
                mask=d.mask, score=float(d.score) * 1e-3,
                bbox=d.bbox, source=d.source, descriptor=d.descriptor,
            ))
        return ordered


# Generic alias for real-scene code. Keep the historical BOP names above because
# all evaluated runs and tests import them.
load_detections = load_bop_detections
DetectionsFileSegmentor = BOPDetectionsSegmentor
