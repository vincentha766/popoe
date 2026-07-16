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


def load_bop_detections(path: str, source: str | None = None) -> list[dict]:
    """Load a BOP-format detections JSON into normalised, numerically-typed
    records, robust to the fully-stringified WA_Sappe (NIDS) variant.

    Each returned record has int scene_id/image_id/category_id, float score,
    and a `segmentation` RLE dict normalised by `_normalize_segmentation`.
    `bbox` (if present) is coerced to a float list. Any OTHER fields on the
    record (e.g. BOP's per-detection `time`) are carried through untouched, so
    the loader can also normalise-and-rewrite without dropping metadata.
    `source` is set from the argument when given (it stamps the origin for
    multi-source union), else from the record's own "source" field when
    present, else left absent (the segmentor buckets those under "_")."""
    with open(path) as f:
        raw = json.load(f)
    records = []
    for d in raw:
        rec = dict(d)                        # keep unknown fields (time, ...)
        rec["scene_id"] = _to_int(d["scene_id"])
        rec["image_id"] = _to_int(d["image_id"])
        rec["category_id"] = _to_int(d["category_id"])
        rec["score"] = _to_float(d["score"])
        rec["segmentation"] = _normalize_segmentation(d["segmentation"])
        if "bbox" in d:
            rec["bbox"] = _coerce_bbox(d["bbox"])
        if source is not None:
            rec["source"] = source
        records.append(rec)
    return records


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
    from pycocotools import mask as coco_mask
    h, w = int(seg["size"][0]), int(seg["size"][1])
    counts = seg["counts"]
    rle = coco_mask.frPyObjects(seg, h, w) if isinstance(counts, list) else seg
    m = coco_mask.decode(rle).astype(bool)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m


class BOPDetectionsSegmentor:
    """Segmentor over a BOP default-detections JSON.

    Args:
        detections_json: path to the BOP-format detections file (list of dicts
            with scene_id / image_id / category_id / score / segmentation RLE,
            optionally a "source" tag when files were unioned).
        topk: detections kept per (source, label) bucket.
        merge_labels: {obj_id: [obj_id, partner_id, ...]} — pool these labels'
            candidates for the given object. Default None (no pooling).
        iou_dedupe: drop a mask whose IoU with an already-kept one exceeds this.
        min_pixels: drop masks smaller than this (unreliable geometry).
    """

    source = "bop-detections"

    def __init__(self, detections_json: str, topk: int = 2,
                 merge_labels: dict | None = None, iou_dedupe: float = 0.9,
                 min_pixels: int = 100):
        self.topk = topk
        self.merge_labels = merge_labels or {}
        self.iou_dedupe = iou_dedupe
        self.min_pixels = min_pixels
        self._by_img: dict = {}
        for d in load_bop_detections(detections_json):
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

        dets: list[Detection] = []
        kept_masks: list[np.ndarray] = []
        for d in sorted(picked, key=lambda d: -d["score"]):
            m = decode_detection_mask(d["segmentation"])
            if m.sum() < self.min_pixels:
                continue
            if any(_mask_iou(m, prev) > self.iou_dedupe for prev in kept_masks):
                continue
            kept_masks.append(m)
            dets.append(Detection(mask=m, score=float(d["score"]),
                                  source=self.source))
        return dets
