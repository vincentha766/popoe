"""Emit BOP `mask_visib` ground truth as a detections JSON.

The point is an upper bound: what does the pipeline reach when perception is
perfect? Rather than teach `bop_eval` a ground-truth mode — which would put a
second, differently-exercised code path next to the evaluated one — this
writes ground-truth masks in the same schema the CNOS/SAM6D files use. The
evaluated pipeline then runs unchanged and the mask SOURCE is the only thing
that differs.

`mask_visib` (visible part), not `mask` (full silhouette), because that is
what a segmentor can produce from one image; scoring against the amodal mask
would bound something no perception module could ever deliver.

Only objects named in the targets file are emitted, so the candidate set
matches what a run is asked to estimate. score is 1.0 and time 0.0: these are
oracle inputs, not a detector's output, and the resulting CSV is a diagnostic
— it is NOT submittable, since the detection stage it implies does not exist.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as cocomask

from popoe.datasets.bop import bop_layout, default_targets_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bop", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--objs", default="")
    args = ap.parse_args()

    bop = Path(args.bop)
    layout = bop_layout(args.dataset)
    targets = json.load(open(default_targets_path(bop, layout["split"])))
    obj_filter = {int(x) for x in args.objs.split(",") if x.strip()}

    wanted = defaultdict(set)
    for t in targets:
        if obj_filter and t["obj_id"] not in obj_filter:
            continue
        wanted[(t["scene_id"], t["im_id"])].add(t["obj_id"])

    import cv2
    out, n_missing, n_empty = [], 0, 0
    for (scene_id, im_id), objs in sorted(wanted.items()):
        sdir = bop / layout["split"] / f"{scene_id:06d}"
        gt = json.load(open(sdir / "scene_gt.json")).get(str(im_id), [])
        for gi, e in enumerate(gt):
            if e["obj_id"] not in objs:
                continue
            p = sdir / "mask_visib" / f"{im_id:06d}_{gi:06d}.png"
            if not p.exists():
                n_missing += 1
                continue
            m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if m is None:
                n_missing += 1
                continue
            m = (m > 0).astype(np.uint8)
            if m.sum() == 0:
                # A fully occluded instance: BOP still lists it, but no
                # segmentor could return it and no depth backs it. Dropping it
                # keeps the bound honest — counted, never silent.
                n_empty += 1
                continue
            rle = cocomask.encode(np.asfortranarray(m))
            ys, xs = np.nonzero(m)
            out.append({
                "scene_id": int(scene_id),
                "image_id": int(im_id),
                "category_id": int(e["obj_id"]),
                "score": 1.0,
                "time": 0.0,
                "bbox": [float(xs.min()), float(ys.min()),
                         float(xs.max() - xs.min() + 1),
                         float(ys.max() - ys.min() + 1)],
                "segmentation": {"size": [int(m.shape[0]), int(m.shape[1])],
                                 "counts": rle["counts"].decode()},
            })
    json.dump(out, open(args.out, "w"))
    print(f"{len(out)} GT detections over {len(wanted)} images -> {args.out}"
          f"  (skipped: {n_missing} missing mask files, {n_empty} fully "
          f"occluded)", flush=True)


if __name__ == "__main__":
    main()
