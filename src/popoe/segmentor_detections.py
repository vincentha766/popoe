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

    def __init__(self, detections_json: str, topk: int = 2,
                 merge_labels: dict | None = None, iou_dedupe: float = 0.9,
                 min_pixels: int = 100):
        self.topk = topk
        self.merge_labels = merge_labels or {}
        self.iou_dedupe = iou_dedupe
        self.min_pixels = min_pixels
        self._by_img: dict = {}
        for d in json.load(open(detections_json)):
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

        from pycocotools import mask as coco_mask
        dets: list[Detection] = []
        kept_masks: list[np.ndarray] = []
        for d in sorted(picked, key=lambda d: -d["score"]):
            seg = d["segmentation"]
            rle = (coco_mask.frPyObjects(seg, seg["size"][0], seg["size"][1])
                   if isinstance(seg["counts"], list) else seg)
            m = coco_mask.decode(rle).astype(bool)
            if m.ndim == 3:
                m = m[:, :, 0]
            if m.sum() < self.min_pixels:
                continue
            if any(_mask_iou(m, prev) > self.iou_dedupe for prev in kept_masks):
                continue
            kept_masks.append(m)
            dets.append(Detection(mask=m, score=float(d["score"])))
        return dets
