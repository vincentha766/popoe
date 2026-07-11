"""BOP evaluation through popoe stages — the framework-validation run.

Reproduces the reproduction study's formal pipeline semantics using popoe's
stage contracts composed at the runner level:

  detections Segmentor (label pooling) -> encode once at w=1
  -> for each (mask, visual weight): solve -> refine -> ChampionScorer
  -> Selector argmax across ALL (mask x weight) hypotheses -> BOP CSV row

Workflow features (the experiment accelerators):

  * ``--cache DIR``  — per-candidate target features persist to disk; reruns
    that only change registration/scoring skip GeDi+DINO entirely.
  * ``--cand-csv F`` — every (mask x weight) hypothesis is dumped with its
    score breakdown, so selection rules can be swapped OFFLINE (zero GPU).
  * Resumable: rows already in --out are skipped on relaunch.

Per-object visual PCA is snapshotted at query encoding and re-installed
before each target encode (install_pca) — the image-major loop interleaves
objects, and the shared fusion instance would otherwise leak one object's
PCA into another's target features.

Usage (pod):
  python examples/bop_eval.py --bop /workspace/bop_data/ycbv \
      --detections /workspace/bop_data/detections/cnos/BOP23/fastSAM_pbr/fastSAM_pbr_ycbv.json \
      --out popoe_ycbv.csv --cache /workspace/popoe_cache_ycbv \
      [--objs 5,8,...] [--weights 1.0,0.7,0.5,0.3,0.2] [--cand-csv cands.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from popoe.adapters import BestScoreSelector
from popoe.interfaces import ObjectModel, PointFeatures, PoseHypothesis, Scene
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
    ap.add_argument("--cache", default="", help="target-feature cache dir")
    ap.add_argument("--cand-csv", default="", help="dump every hypothesis")
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
    if args.cache:
        os.makedirs(args.cache, exist_ok=True)

    targets = json.load(open(bop / "test_targets_bop19.json"))
    if obj_filter:
        targets = [t for t in targets if t["obj_id"] in obj_filter]
    by_img: dict = {}
    for t in targets:
        by_img.setdefault((t["scene_id"], t["im_id"]), []).append(t["obj_id"])

    # Resume: skip targets already written.
    done: set = set()
    if os.path.exists(args.out):
        for r in csv.DictReader(open(args.out)):
            done.add((int(r["scene_id"]), int(r["im_id"]), int(r["obj_id"])))
    print(f"{len(targets)} targets / {len(by_img)} images"
          + (f" (resuming past {len(done)})" if done else ""), flush=True)

    segmentor = best_segmentor(args.detections, topk=args.topk, merge_labels=merge)
    q_enc, t_enc = best_encoders(target_grid=args.grid)
    selector = BestScoreSelector()

    # Encode ALL queries up front (fail fast; PCA snapshots live in meta).
    # Query features + fitted PCA are CACHED alongside target features: the
    # cached targets' visual halves are projected in the query PCA basis, so
    # re-fitting the PCA in a later run silently breaks them (PCA signs are
    # arbitrary per fit — see fusion.py). One basis per object, persisted.
    import pickle
    query_cache: dict = {}
    for obj_id in sorted({o for objs in by_img.values() for o in objs}):
        obj = ObjectModel(obj_id=obj_id,
                          mesh_path=str(bop / "models" / f"obj_{obj_id:06d}.ply"),
                          diameter=0.0)
        t0 = time.time()
        qnpz = os.path.join(args.cache, f"query_{obj_id}.npz") if args.cache else None
        qpkl = os.path.join(args.cache, f"query_{obj_id}_pca.pkl") if args.cache else None
        if qnpz and os.path.exists(qnpz) and os.path.exists(qpkl):
            z = np.load(qnpz)
            q = PointFeatures(pts=z["pts"], feats=z["feats"],
                              meta={"pca_vis": pickle.load(open(qpkl, "rb"))})
            from popoe.interfaces import CanonFrame
            q.meta["canon_frame"] = CanonFrame.from_points(q.pts)
        else:
            q = q_enc.encode_query(obj)
            if qnpz:
                np.savez_compressed(qnpz, pts=q.pts, feats=q.feats)
                pickle.dump(q.meta.get("pca_vis"), open(qpkl, "wb"))
        q.meta["feats_w1"] = q.feats
        extent = float(np.ptp(q.pts, axis=0).max())
        stages = stages_for_object(extent, size_aware=obj_id in merge)
        query_cache[obj_id] = (obj, q, stages)
        print(f"  obj{obj_id}: extent={extent*1000:.0f}mm "
              f"encode={time.time()-t0:.1f}s", flush=True)

    cand_f = None
    if args.cand_csv:
        new = not os.path.exists(args.cand_csv)
        cand_f = open(args.cand_csv, "a", newline="")
        cand_wr = csv.writer(cand_f)
        if new:
            cand_wr.writerow(["scene_id", "im_id", "obj_id", "cand", "w",
                              "s_icp", "s_feat_1", "metric_fit", "score",
                              "R", "t"])

    def encode_target_cached(scene, det, obj, q, ci):
        key = (f"{scene.scene_id}_{scene.im_id}_{obj.obj_id}_{ci}.npz"
               if args.cache else None)
        if key and os.path.exists(os.path.join(args.cache, key)):
            z = np.load(os.path.join(args.cache, key))
            return PointFeatures(pts=z["pts"], feats=z["feats"],
                                 meta={"feats_w1": z["feats"]})
        t_enc.install_pca(q.meta.get("pca_vis"))
        tgt = t_enc.encode_target(scene, det, obj, q.meta.get("canon_frame"))
        if len(tgt.pts) >= 4 and key:
            np.savez_compressed(os.path.join(args.cache, key),
                                pts=tgt.pts, feats=tgt.feats)
        tgt.meta["feats_w1"] = tgt.feats
        return tgt

    n_done = 0
    header_needed = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        wr = csv.writer(f)
        if header_needed:
            wr.writerow(["scene_id", "im_id", "obj_id", "score", "R", "t", "time"])
        for (scene_id, im_id), obj_ids in sorted(by_img.items()):
            pending = [o for o in obj_ids if (scene_id, im_id, o) not in done]
            if not pending:
                continue
            sdir = bop / "test" / f"{scene_id:06d}"
            cam = json.load(open(sdir / "scene_camera.json"))[str(im_id)]
            K = np.array(cam["cam_K"]).reshape(3, 3)
            depth_raw = cv2.imread(str(sdir / "depth" / f"{im_id:06d}.png"),
                                   cv2.IMREAD_UNCHANGED)
            rgb = cv2.cvtColor(cv2.imread(str(sdir / "rgb" / f"{im_id:06d}.png")),
                               cv2.COLOR_BGR2RGB)
            if depth_raw is None or rgb is None:
                for o in pending:
                    wr.writerow([scene_id, im_id, o, 0.0, IDN, ZT, "0.0"])
                continue
            depth = depth_raw.astype(np.float32) * cam["depth_scale"] / 1000.0
            scene = Scene(rgb=rgb, depth=depth, K=K,
                          scene_id=scene_id, im_id=im_id)

            for obj_id in pending:
                t_start = time.time()
                obj, q, (solver, refiner, scorer) = query_cache[obj_id]
                frame = q.meta.get("canon_frame")
                hyps: list[PoseHypothesis] = []
                try:
                    dets = segmentor.segment(scene, obj)
                except Exception:
                    dets = []
                for ci, det in enumerate(dets):
                    try:
                        tgt = encode_target_cached(scene, det, obj, q, ci)
                    except Exception:
                        continue        # degenerate mask cloud (e.g. GeDi LRF)
                    if len(tgt.pts) < 4:
                        continue
                    for w in weights:
                        qw = q if w == 1.0 else _reweighted(q, w)
                        tw = tgt if w == 1.0 else _reweighted(tgt, w)
                        try:
                            for h in solver.solve(qw, tw, frame):
                                h = refiner.refine(h, scene, obj, qw, tw)
                                h = scorer.score(h, qw, tw)
                                hyps.append(h)
                                if cand_f is not None:
                                    cand_wr.writerow([
                                        scene_id, im_id, obj_id, ci, w,
                                        f"{h.breakdown.get('s_icp', 0):.4f}",
                                        f"{h.breakdown.get('s_feat_1', 0):.4f}",
                                        f"{h.breakdown.get('metric_fit', 1):.4f}",
                                        f"{h.score:.6f}",
                                        " ".join(f"{v:.6f}" for v in h.R.flatten()),
                                        " ".join(f"{v:.4f}" for v in (h.t * 1000.0))])
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
                f.flush()
                n_done += 1
                if n_done % 50 == 0:
                    print(f"{n_done} targets this run", flush=True)
    if cand_f is not None:
        cand_f.close()
    print(f"done -> {args.out}", flush=True)


def _reweighted(pf, w: float):
    return PointFeatures(pts=pf.pts, feats=scale_vis(pf.meta["feats_w1"], w),
                         pts_dense=pf.pts_dense, meta=pf.meta)


if __name__ == "__main__":
    main()
