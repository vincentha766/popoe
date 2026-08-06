"""Does the existing score prefer a flipped pose when offered one?

The seven-set analysis found that the failing half of a wrong pose is almost
always the orientation, and that the wrong orientations concentrate on
objects whose GEOMETRY is rotationally ambiguous: a clean 180 deg flip about
the mustard bottle's long axis (all five visual weights agree on it), and a
flat azimuth histogram on the cylindrical cans. Meanwhile the winning
rotation is chosen inside Open3D's RANSAC by geometric inlier count alone —
the visual features only re-rank the ten survivors.

So: if we hand the scorer the flipped counterparts explicitly, does it pick
the better one? The scorer never sees ground truth here; it ranks with the
same s_icp * max(s_feat_1, 0) it always uses. Ground truth enters only
afterwards, when this run's output is compared against it.

Both outcomes are informative. If the score prefers the correct flip, the
visual branch could have disambiguated all along and simply never voted on
rotation — a selection fix. If it does not, the ambiguity is beyond what
these features can resolve, which is a property of the benchmark rather than
a defect to repair.

Every candidate goes through the identical refine-then-score path, including
the unflipped champion, so no variant gets an unfair advantage. Features come
from the campaign cache; a miss is fatal rather than silently re-encoded,
because re-encoding here would not match what produced the champion.

Usage (pod, fresh clone):
  python scripts/flip_rescore_ab.py \
      --bop /workspace/bop_data/ycbv --dataset ycbv \
      --detections .../fastSAM_pbr_ycbv.json \
      --cand-csv .../ycbv_cands.csv \
      --cache /workspace/results/pipeline_verify_20260726/cache_ycbv_g32m \
      --out flip_ab_ycbv.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from popoe.cache import StageCache, file_fingerprint, fingerprint
from popoe.datasets.bop import bop_layout, default_targets_path
from popoe.freeze.recipes import (YCBV_MERGE_LABELS, best_segmentor,
                                  stages_for_object)
from popoe.interfaces import (CanonFrame, ObjectModel, PointFeatures,
                              PoseHypothesis, Scene)


def slice_cosines(R, t, pts_q, pts_t, fq, ft, tau):
    """Cosine sums over the SAME inlier set for the fused vector and for its
    visual / geometric halves separately.

    The fused vector is `concat([1.0 * l2norm(vis), l2norm(geo)])` at w=1
    (fusion.py), so each half already carries unit norm and the fused cosine
    is EXACTLY the mean of the two half cosines. For a flip on a
    geometrically ambiguous object the geometric half barely moves, so it
    contributes half the similarity while carrying almost none of the signal.
    Splitting the halves is what tells us whether the visual features hold a
    preference that the fused average is drowning.

    The inlier set is geometric, hence identical across the three sums —
    they differ only in which dimensions are compared.
    """
    from sklearn.neighbors import KDTree
    tq = (R @ pts_q.T).T + t
    d, nn = KDTree(tq).query(pts_t, k=1)
    inl = d[:, 0] < tau
    n_inl = int(inl.sum())
    if n_inl == 0:
        return 0.0, 0.0, 0.0, 0
    A, B = ft[inl], fq[nn[:, 0][inl]]
    h = A.shape[1] // 2

    def csum(x, y):
        xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        yn = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-8)
        return float((xn * yn).sum())

    return csum(A, B), csum(A[:, :h], B[:, :h]), csum(A[:, h:], B[:, h:]), n_inl


def rot_about(axis, deg):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    th = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K.dot(K)


def variants(R, pts):
    """Champion plus rotational alternates, expressed in the MODEL frame so a
    flip means 'the same object turned around', not 'the camera moved'.

    Axes come from the query cloud's principal directions: the ambiguities
    observed are about an object's own long axis (bottle flip) and about a
    cylinder's symmetry axis, and PCA recovers both without ground truth.
    """
    c = pts - pts.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    out = [("champion", R)]
    for i in range(3):
        out.append((f"flip{i}", R.dot(rot_about(vt[i], 180.0))))
    for ang in (90.0, 270.0):
        out.append((f"az{int(ang)}", R.dot(rot_about(vt[0], ang))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bop", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--detections", required=True)
    ap.add_argument("--cand-csv", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--objs", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    bop = Path(args.bop)
    layout = bop_layout(args.dataset)
    merge = dict(YCBV_MERGE_LABELS) if args.dataset == "ycbv" else {}
    cache = StageCache(args.cache)

    # Must reproduce bop_eval's enc_cfg EXACTLY or every cache lookup misses.
    # render_backend is the string the campaign ran under; it is spelled out
    # rather than obtained from an encoder so this script never loads a model
    # and can run on a CPU-only box.
    enc_cfg = {
        "grid": os.environ.get("POPOE_TARGET_GRID", "32"),
        "n_points": 3000,
        "dino_layer": os.environ.get("POPOE_DINO_LAYER", "ratio0.78"),
        "two_scale": os.environ.get("POPOE_TWO_SCALE_GEDI", "1"),
        "crop": os.environ.get("POPOE_TARGET_CROP", "1"),
        "vis_dim": os.environ.get("POPOE_VIS_DIM", "geo-matched"),
        "n_views": os.environ.get("POPOE_N_VIEWS", "162"),
        "target_fill": os.environ.get("POPOE_TARGET_FILL", "0.5"),
        "target_canon": os.environ.get("POPOE_TARGET_CANON", "224"),
        "vis_weight": "1.0-pinned",
        "skip_vis": os.environ.get("POPOE_SKIP_VIS", "0"),
        "geom_backbone": os.environ.get("POPOE_GEOM_BACKBONE", "gedi"),
        "dgedi_mode": os.environ.get("POPOE_DGEDI_MODE", "single_scale"),
        "gedi_path": os.environ.get("POPOE_GEDI_PATH", "/workspace/gedi"),
        "render_backend": "nvdiffrast",
    }
    # The same conditional knobs bop_eval adds (paper-fidelity features + the
    # viewpoint layout): each enters the key ONLY at a non-default value. This
    # mirror drifted once already — the faithful cache's keys carry these and
    # this script's replica did not, so every lookup missed. If bop_eval grows
    # another knob, it must land here too, which is the cost of mirroring.
    for _key, (_env, _dflt) in {
        "query_canon": ("POPOE_QUERY_CANON", "224"),
        "query_fill": ("POPOE_QUERY_FILL", "0.45"),
        "query_min_views": ("POPOE_QUERY_MIN_VIEWS", "0"),
        "canon_basis": ("POPOE_CANON_BASIS", "extent"),
        "query_views": ("POPOE_QUERY_VIEWS", "spiral"),
    }.items():
        _val = os.environ.get(_env, _dflt)
        if _val != _dflt:
            enc_cfg[_key] = _val
    enc_cfg["n_points"] = int(os.environ.get("POPOE_QUERY_POINTS", "3000"))

    champs = {}
    for r in csv.DictReader(open(args.cand_csv)):
        k = (int(r["scene_id"]), int(r["im_id"]), int(r["obj_id"]))
        s = float(r["score"])
        if k not in champs or s > champs[k]["score"]:
            champs[k] = dict(
                score=s, cand=int(r["cand"]), w=float(r["w"]),
                R=np.array([float(x) for x in r["R"].split()]).reshape(3, 3),
                t=np.array([float(x) for x in r["t"].split()], float) / 1000.0)

    targets = json.load(open(default_targets_path(bop, layout["split"])))
    obj_filter = {int(x) for x in args.objs.split(",") if x.strip()}
    keys = sorted({(t["scene_id"], t["im_id"], t["obj_id"]) for t in targets
                   if not obj_filter or t["obj_id"] in obj_filter})
    if args.limit:
        keys = keys[:args.limit]

    segmentor = best_segmentor(args.detections, topk=2, merge_labels=merge)

    qcache = {}
    def query_of(obj_id):
        if obj_id in qcache:
            return qcache[obj_id]
        mesh = str(bop / layout["models_dir"] / f"obj_{obj_id:06d}.ply")
        # Shading key parts, same as bop_eval: LM-O-family meshes carry
        # `shading=vcolor-v1` since the vertex-colour fix, and this mirror
        # missing it is exactly why every post-fix LM-O lookup missed (the
        # second mirror-drift failure in one night; the first was the
        # conditional knobs). The resolver is pure CPU by design.
        from popoe.freeze.feature_extractor import mesh_shading_key_parts
        shade_parts = mesh_shading_key_parts(mesh)
        qkey = fingerprint("query", enc_cfg, file_fingerprint(mesh), obj_id,
                           *shade_parts)
        arrays = cache.get_arrays("query", qkey)
        pca = cache.get_pickle("query", qkey)
        if arrays is None or pca is None:
            raise SystemExit(
                f"query cache miss for obj {obj_id} (key {qkey}). The cache "
                f"must be the one the campaign wrote, and enc_cfg must match "
                f"it exactly — re-encoding here would not reproduce the "
                f"features that produced the champion.")
        q = PointFeatures(pts=arrays["pts"], feats=arrays["feats"],
                          meta={"pca_vis": pca, "qkey": qkey,
                                "feats_w1": arrays["feats"]})
        # B1-F1: serve the BUILD-TIME scale. from_points() is only the
        # historical convention for extent-basis, ungated caches; a faithful
        # campaign cache (diameter + MIN_VIEWS=18) stores a filtered cloud
        # whose extent is NOT the recorded basis.
        if "canon_scale" in arrays:
            q.meta["canon_frame"] = CanonFrame(center=np.zeros(3, np.float32),
                                               scale=float(arrays["canon_scale"]))
        elif (os.environ.get("POPOE_CANON_BASIS", "extent") == "extent"
              and os.environ.get("POPOE_QUERY_MIN_VIEWS", "0") == "0"):
            q.meta["canon_frame"] = CanonFrame.from_points(q.pts)
        else:
            raise SystemExit(
                f"query cache entry for obj {obj_id} predates the canon_scale "
                f"record and the configured basis/visibility gate cannot be "
                f"reconstructed from the stored points (B1-F1). Re-encode the "
                f"campaign cache with current code before rescoring.")
        obj = ObjectModel(obj_id=obj_id, mesh_path=mesh, diameter=0.0)
        extent = float(np.ptp(q.pts, axis=0).max())
        stages = stages_for_object(extent, size_aware=obj_id in merge,
                                   solver="o3d")
        qcache[obj_id] = (obj, q, stages)
        return qcache[obj_id]

    by_img = defaultdict(list)
    for s, i, o in keys:
        by_img[(s, i)].append(o)

    n_rows = n_miss = n_done = 0
    with open(args.out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["scene_id", "im_id", "obj_id", "variant", "score",
                     "s_icp", "s_feat_1", "n_inliers", "n_target", "cos_sum",
                     "cos_sum_vis", "cos_sum_geo", "R", "t"])
        for (scene_id, im_id), objs in sorted(by_img.items()):
            sdir = bop / layout["split"] / f"{scene_id:06d}"
            cam = json.load(open(sdir / "scene_camera.json"))[str(im_id)]
            K = np.array(cam["cam_K"]).reshape(3, 3)
            depth_raw = cv2.imread(
                str(sdir / "depth" / f"{im_id:06d}{layout['depth_ext']}"),
                cv2.IMREAD_UNCHANGED)
            bgr = cv2.imread(
                str(sdir / layout["img_dir"] / f"{im_id:06d}{layout['img_ext']}"))
            if depth_raw is None or bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth = depth_raw.astype(np.float32) * cam["depth_scale"] / 1000.0
            scene = Scene(rgb=rgb, depth=depth, K=K, scene_id=scene_id,
                          im_id=im_id)
            scene_fp = fingerprint(rgb, depth, K)

            for obj_id in sorted(objs):
                key = (scene_id, im_id, obj_id)
                if key not in champs:
                    continue
                ch = champs[key]
                obj, q, (_solver, refiner, scorer) = query_of(obj_id)
                dets = segmentor.segment(scene, obj)
                if ch["cand"] >= len(dets):
                    n_miss += 1
                    continue
                det = dets[ch["cand"]]
                tkey = fingerprint("target", enc_cfg, scene_fp, obj_id,
                                   det.mask, q.meta["qkey"])
                hit = cache.get_arrays("target", tkey)
                if hit is None:
                    n_miss += 1
                    continue
                tgt = PointFeatures(pts=hit["pts"], feats=hit["feats"],
                                    meta={"feats_w1": hit["feats"]})
                # tau exactly as ChampionScorer derives it (scoring.py), so the
                # re-derived ingredients describe the same inlier set the live
                # score used — not a near-miss reimplementation.
                tau = 0.03 * float(np.ptp(q.pts, axis=0).max())
                for name, Rv in variants(ch["R"], q.pts):
                    h = PoseHypothesis(R=Rv, t=ch["t"], score=0.0)
                    h = refiner.refine(h, scene, obj, q, tgt)
                    h = scorer.score(h, q, tgt)
                    # Raw ingredients, so any normalisation can be compared
                    # offline without another pod trip: the live score divides
                    # the cosine sum by |I|, the paper's Eq.5 divides by the
                    # fixed |P_T|, and both are recoverable from these.
                    c_all, c_vis, c_geo, n_inl = slice_cosines(
                        h.R, h.t, q.pts, tgt.pts,
                        q.meta["feats_w1"], tgt.meta["feats_w1"], tau)
                    wr.writerow([scene_id, im_id, obj_id, name,
                                 f"{h.score:.6f}",
                                 f"{h.breakdown.get('s_icp', 0):.4f}",
                                 f"{h.breakdown.get('s_feat_1', 0):.4f}",
                                 n_inl, len(tgt.pts),
                                 f"{c_all:.6f}", f"{c_vis:.6f}", f"{c_geo:.6f}",
                                 " ".join(f"{v:.6f}" for v in h.R.flatten()),
                                 " ".join(f"{v:.4f}" for v in (h.t * 1000.0))])
                    n_rows += 1
                n_done += 1
                if n_done % 200 == 0:
                    print(f"{n_done} targets, {n_rows} rows, {n_miss} skipped",
                          flush=True)
    print(f"done: {n_done} targets, {n_rows} rows, {n_miss} skipped "
          f"(cache/detection mismatch) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
