"""BOP evaluation through popoe stages — the framework-validation run.

Reproduces the reproduction study's formal pipeline semantics using popoe's
stage contracts composed at the runner level:

  detections Segmentor (label pooling) -> encode once at w=1
  -> for each (mask, visual weight): solve -> refine -> ChampionScorer
  -> Selector argmax across ALL (mask x weight) hypotheses -> BOP CSV row

The per-target weight sweep is why this composes stages manually instead of
calling interfaces.Pipeline.run (which is single-weight): features extract
ONCE and the visual half is rescaled per weight (recipes.scale_vis).

Usage (pod):
  python examples/bop_eval.py --bop /workspace/bop_data/ycbv \
      --detections /workspace/bop_data/detections/cnos/BOP23/fastSAM_pbr/fastSAM_pbr_ycbv.json \
      --out popoe_ycbv.csv [--objs 5,8,10,14,17,19,20,21] [--weights 1.0,0.7,0.5,0.3,0.2]
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from popoe.adapters import BestScoreSelector
from popoe.interfaces import ObjectModel, PoseHypothesis, Scene
from popoe.recipes import (WEIGHTS, YCBV_MERGE_LABELS, best_encoders,
                           best_segmentor, scale_vis, stages_for_object)

IDN = " ".join(f"{v:.6f}" for v in np.eye(3).flatten())
ZT = "0.0 0.0 0.0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bop", required=True)
    ap.add_argument("--detections", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--objs", default="")
    ap.add_argument("--weights", default=",".join(str(w) for w in WEIGHTS))
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--merge", default="ycbv",
                    help="'ycbv' for the clamp pair, 'none', or '19:20,...'")
    args = ap.parse_args()

    bop = Path(args.bop)
    weights = [float(w) for w in args.weights.split(",")]
    obj_filter = {int(x) for x in args.objs.split(",") if x.strip()}
    if args.merge == "ycbv":
        merge = YCBV_MERGE_LABELS
    elif args.merge == "none":
        merge = {}
    else:
        merge = {}
        for grp in args.merge.split(","):
            ids = [int(x) for x in grp.split(":")]
            for a in ids:
                merge[a] = ids

    targets = json.load(open(bop / "test_targets_bop19.json"))
    if obj_filter:
        targets = [t for t in targets if t["obj_id"] in obj_filter]
    by_img: dict = {}
    for t in targets:
        by_img.setdefault((t["scene_id"], t["im_id"]), []).append(t["obj_id"])
    print(f"{len(targets)} targets / {len(by_img)} images", flush=True)

    segmentor = best_segmentor(args.detections, topk=args.topk, merge_labels=merge)
    q_enc, t_enc = best_encoders(target_grid=args.grid)
    selector = BestScoreSelector()

    # Per-object: query features (w=1), per-object stages, model metadata.
    query_cache: dict = {}

    def get_query(obj_id: int):
        if obj_id not in query_cache:
            obj = ObjectModel(obj_id=obj_id,
                              mesh_path=str(bop / "models" / f"obj_{obj_id:06d}.ply"),
                              diameter=0.0)
            t0 = time.time()
            q = q_enc.encode_query(obj)
            q.meta["feats_w1"] = q.feats
            extent = float(np.ptp(q.pts, axis=0).max())
            stages = stages_for_object(extent, size_aware=obj_id in merge)
            query_cache[obj_id] = (obj, q, stages)
            print(f"  obj{obj_id}: extent={extent*1000:.0f}mm "
                  f"encode={time.time()-t0:.1f}s", flush=True)
        return query_cache[obj_id]

    n_done = 0
    with open(args.out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["scene_id", "im_id", "obj_id", "score", "R", "t", "time"])
        for (scene_id, im_id), obj_ids in sorted(by_img.items()):
            sdir = bop / "test" / f"{scene_id:06d}"
            cam = json.load(open(sdir / "scene_camera.json"))[str(im_id)]
            K = np.array(cam["cam_K"]).reshape(3, 3)
            depth_raw = cv2.imread(str(sdir / "depth" / f"{im_id:06d}.png"),
                                   cv2.IMREAD_UNCHANGED)
            rgb = cv2.cvtColor(cv2.imread(str(sdir / "rgb" / f"{im_id:06d}.png")),
                               cv2.COLOR_BGR2RGB)
            if depth_raw is None or rgb is None:
                for o in obj_ids:
                    wr.writerow([scene_id, im_id, o, 0.0, IDN, ZT, "0.0"])
                continue
            depth = depth_raw.astype(np.float32) * cam["depth_scale"] / 1000.0
            scene = Scene(rgb=rgb, depth=depth, K=K,
                          scene_id=scene_id, im_id=im_id)

            for obj_id in obj_ids:
                t_start = time.time()
                obj, q, (solver, refiner, scorer) = get_query(obj_id)
                frame = q.meta.get("canon_frame")
                hyps: list[PoseHypothesis] = []
                for det in segmentor.segment(scene, obj):
                    tgt = t_enc.encode_target(scene, det, obj, frame)
                    if len(tgt.pts) < 4:
                        continue
                    tgt.meta["feats_w1"] = tgt.feats
                    for w in weights:
                        qw = q if w == 1.0 else _reweighted(q, w)
                        tw = tgt if w == 1.0 else _reweighted(tgt, w)
                        try:
                            for h in solver.solve(qw, tw, frame):
                                h = refiner.refine(h, scene, obj, qw, tw)
                                hyps.append(scorer.score(h, qw, tw))
                        except Exception:
                            continue
                best = selector.select(hyps)
                if best is None:
                    wr.writerow([scene_id, im_id, obj_id, 0.0, IDN, ZT,
                                 f"{time.time()-t_start:.3f}"])
                else:
                    wr.writerow([scene_id, im_id, obj_id, f"{best.score:.6f}",
                                 " ".join(f"{v:.6f}" for v in best.R.flatten()),
                                 " ".join(f"{v:.4f}" for v in (best.t * 1000.0)),
                                 f"{time.time()-t_start:.3f}"])
                n_done += 1
                if n_done % 50 == 0:
                    print(f"{n_done}/{len(targets)}", flush=True)
    print(f"done -> {args.out}", flush=True)


def _reweighted(pf, w: float):
    """Copy of PointFeatures with the visual half rescaled; keeps meta (incl.
    feats_w1 for the canonical-space scorer)."""
    from popoe.interfaces import PointFeatures
    return PointFeatures(pts=pf.pts, feats=scale_vis(pf.meta["feats_w1"], w),
                         pts_dense=pf.pts_dense, meta=pf.meta)


if __name__ == "__main__":
    main()
