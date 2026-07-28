"""Per-view matching probe: is the view AVERAGE what kills the pairings?

The banana's visual-half pairings are almost all wrong (rate1 ~ 0.01) while
image-to-image comparison with the same DINOv2 tells its orientations apart.
One suspect is the aggregation: a query point's stored feature is the MEAN of
its ~60 visible views' features, and the photo only ever shows one view. This
probe re-extracts ONE object's query features keeping every per-view vector,
then scores the same GT top-1 check under three matching rules:

    mean       average over views, renormalised  (reproduces the pipeline)
    maxview    sim(t, q) = max over views of cos(f_t, f_{q,v})
    retrieval  pick the single render whose pooled masked-patch feature best
               matches the crop's pooled feature, then match within that view
               (FoundPose's shape, and the cheap variant to implement live)

Visual half only: the geometric half is view-independent by construction and
would only dilute the contrast this probe is after. Target features come from
the pipeline's own TargetFeatureExtractor (no cache-key mirroring); the query
view loop is COPIED from QueryFeatureExtractor.extract_query_features (camera
spiral, raycast, bilinear patch sampling) — if that loop changes, this copy
must follow. PCA basis: fitted here on the mean features (per-object fit, the
same procedure the pipeline uses), applied identically to all three rules, so
the comparison is internal and self-consistent.

Usage (pod, GPU):
  python scripts/perview_probe.py --bop /workspace/bop_data/ycbv \
      --dataset ycbv --obj 10 \
      --detections .../cnos-fastsam_ycbv-test.json \
      --out /workspace/results/perview_banana.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("POPOE_BOP_TOOLKIT", "/workspace/bop_toolkit"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bop", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--obj", type=int, required=True)
    ap.add_argument("--detections", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-points", type=int, default=3000)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    import trimesh
    import cv2
    from torchvision import transforms
    from bop_toolkit_lib import misc as btk_misc

    from popoe.freeze.feature_extractor import (
        QueryFeatureExtractor, TargetFeatureExtractor, load_dinov2, load_gedi,
        _dino_layer)
    from popoe.freeze.recipes import best_segmentor
    from popoe.interfaces import CanonFrame, ObjectModel, PointFeatures, Scene
    from popoe.datasets.bop import bop_layout
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from examples.bop_eval import probe_corr_stats, read_frame_images

    bop = Path(args.bop)
    layout = bop_layout(args.dataset)
    oid = args.obj
    mesh_path = str(bop / layout["models_dir"] / f"obj_{oid:06d}.ply")

    dino = load_dinov2("cuda")
    gedi = load_gedi("cuda")
    qx = QueryFeatureExtractor("cuda", dino=dino, gedi=gedi,
                               render_backend="nvdiffrast")
    tx = TargetFeatureExtractor("cuda", dino=dino, gedi=gedi)
    qx.fusion.vis_weight = 1.0
    # The ADAPTER owns install_pca/encode_target; the raw extractor does not
    # (first launch died on exactly this).
    from popoe.freeze.adapters import make_freeze_encoders
    _, t_enc = make_freeze_encoders(qx, tx, n_points=args.n_points)

    # ---- query side: the extract_query_features view loop, per-view kept ----
    # (copied from feature_extractor.extract_query_features; keep in step)
    mesh = trimesh.load(mesh_path, force="mesh")
    pts, _ = trimesh.sample.sample_surface_even(mesh, args.n_points, seed=oid)
    pts_m = (pts / 1000.0).astype(np.float32)          # BOP mm -> m

    canon = (int(os.environ.get("POPOE_QUERY_CANON", "224")) // 14) * 14
    fill = float(os.environ.get("POPOE_QUERY_FILL", "0.45"))
    rmesh = trimesh.load(mesh_path, force="mesh")
    scale = fill * canon / max(rmesh.extents)
    rmesh.apply_scale(scale)
    center = rmesh.bounds.mean(0)
    rmesh.apply_translation(-center)
    mesh_pts = (pts_m * 1000.0 * scale) - center

    n_views = int(os.environ.get("POPOE_N_VIEWS", "162"))
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    H = W = canon
    golden = (1 + math.sqrt(5)) / 2
    radius_cam = max(rmesh.extents) * 1.5

    DINO_DIM = 1536
    per_view = np.full((args.n_points, n_views, DINO_DIM), np.nan, np.float32)
    view_pool = np.zeros((n_views, DINO_DIM), np.float64)   # masked-mean per render
    with torch.no_grad():
        for i in range(n_views):
            theta = math.acos(1 - 2 * (i + 0.5) / n_views)
            phi = 2 * math.pi * i / golden
            cam_pos = radius_cam * np.array([
                math.sin(theta) * math.cos(phi),
                math.sin(theta) * math.sin(phi), math.cos(theta)])
            img_pil, depth_render, fx, fy, cx, cy = qx._raycast_render(
                rmesh, cam_pos, H, W)
            img_t = tfm(img_pil).unsqueeze(0).to("cuda")
            fo = dino.get_intermediate_layers(
                img_t, n=[_dino_layer(dino)], return_class_token=False)[0]
            n_ph, n_pw = H // 14, W // 14
            fmap = fo[0].reshape(n_ph, n_pw, -1).cpu().numpy()

            fwd = -cam_pos / np.linalg.norm(cam_pos)
            up_ref = (np.array([0., 1., 0.]) if abs(fwd[1]) < 0.9
                      else np.array([1., 0., 0.]))
            right = np.cross(fwd, up_ref); right /= np.linalg.norm(right)
            up = np.cross(right, fwd)
            R_cw = np.stack([right, up, fwd], axis=0)
            pts_cam = (R_cw @ (mesh_pts - cam_pos).T).T
            visible = pts_cam[:, 2] > 0
            u = (pts_cam[visible, 0] / pts_cam[visible, 2]) * fx + cx
            v = (pts_cam[visible, 1] / pts_cam[visible, 2]) * fy + cy
            v_int = np.clip(v.astype(int), 0, H - 1)
            u_int = np.clip(u.astype(int), 0, W - 1)
            rd = depth_render[v_int, u_int]
            az = pts_cam[visible, 2]
            tol = 0.05 * float(np.ptp(mesh_pts, axis=0).max())
            ok = (rd > 0) & (np.abs(rd - az) < tol)
            vis_idx = np.where(visible)[0][ok]
            up_ = u[ok] * n_pw / W - 0.5
            vp_ = v[ok] * n_ph / H - 0.5
            i0 = np.clip(np.floor(vp_).astype(int), 0, n_ph - 1)
            j0 = np.clip(np.floor(up_).astype(int), 0, n_pw - 1)
            i1 = np.clip(i0 + 1, 0, n_ph - 1); j1 = np.clip(j0 + 1, 0, n_pw - 1)
            di = np.clip(vp_ - i0, 0, 1)[:, None]
            dj = np.clip(up_ - j0, 0, 1)[:, None]
            samp = ((1 - di) * (1 - dj)) * fmap[i0, j0] \
                + ((1 - di) * dj) * fmap[i0, j1] \
                + (di * (1 - dj)) * fmap[i1, j0] + (di * dj) * fmap[i1, j1]
            per_view[vis_idx, i] = samp
            dm = depth_render > 0
            if dm.any():
                ii, jj = np.where(dm)
                view_pool[i] = fmap[np.clip(ii // 14, 0, n_ph - 1),
                                    np.clip(jj // 14, 0, n_pw - 1)].mean(0)

    seen = ~np.isnan(per_view[:, :, 0])
    print(f"query: {args.n_points} pts, views/pt median "
          f"{int(np.median(seen.sum(1)))}", flush=True)

    # PCA fitted on the MEAN features (the pipeline's own procedure), then
    # applied to every per-view vector so all three rules share one basis.
    mean_raw = np.nanmean(per_view, axis=1)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=64).fit(mean_raw)

    def proj(x):
        y = pca.transform(x)
        n = np.linalg.norm(y, axis=-1, keepdims=True)
        return y / np.maximum(n, 1e-12)

    q_mean = proj(mean_raw)                              # (Nq, 64)
    view_pool_p = proj(view_pool)                        # (V, 64)

    # ---- target side: the pipeline's own extractor, visual half sliced ----
    obj = ObjectModel(obj_id=oid, mesh_path=mesh_path, diameter=0.0)
    seg = best_segmentor(detections_json=args.detections, topk=2)
    t_enc.install_pca(pca)

    mi = json.load(open(bop / layout["models_dir"] / "models_info.json"))
    diam_m = float(mi[str(oid)]["diameter"]) / 1000.0
    syms = btk_misc.get_symmetry_transformations(mi[str(oid)], 0.01)

    targets = json.load(open(bop / "test_targets_bop19.json"))
    todo = [(t["scene_id"], t["im_id"]) for t in targets if t["obj_id"] == oid]
    if args.limit:
        todo = todo[:args.limit]
    gt_cache = {}
    rows = []
    for k, (sid, im) in enumerate(sorted(set(todo))):
        sdir = bop / layout["split"] / f"{sid:06d}"
        cams = json.load(open(sdir / "scene_camera.json"))
        K = np.array(cams[str(im)]["cam_K"]).reshape(3, 3)
        rgb_bgr, depth_raw = read_frame_images(sdir, im, layout)
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        depth = depth_raw.astype(np.float32) * cams[str(im)]["depth_scale"] / 1000.0
        scene = Scene(rgb=rgb, depth=depth, K=K, scene_id=sid, im_id=im)
        if sid not in gt_cache:
            gt_cache[sid] = json.load(open(sdir / "scene_gt.json"))
        gts = [dict(R=np.array(g["cam_R_m2c"]).reshape(3, 3),
                    t=np.array(g["cam_t_m2c"], dtype=float))
               for g in gt_cache[sid].get(str(im), []) if g["obj_id"] == oid]
        if not gts:
            continue
        try:
            dets = seg.segment(scene, obj)
        except Exception:
            continue
        for ci, det in enumerate(dets):
            try:
                tgt = t_enc.encode_target(scene, det, obj,
                                          CanonFrame.from_points(pts_m))
            except Exception:
                continue
            if len(tgt.pts) < 4:
                continue
            # visual half of the fused target features, renormalised
            h = tgt.feats.shape[1] // 2
            tv = tgt.feats[:, :h]
            tv = tv / np.maximum(np.linalg.norm(tv, axis=1, keepdims=True), 1e-12)
            tq = PointFeatures(pts=tgt.pts, feats=tv)

            def r1_of(qfeats):
                qp = PointFeatures(pts=pts_m, feats=qfeats)
                r1, _, reach, _, _ = probe_corr_stats(qp, tq, gts, syms, diam_m)
                return r1, reach

            r1_mean, reach_mean = r1_of(q_mean)

            # maxview: sim (Nt, Nq) = max over views. Chunk over views.
            sim_best = np.full((len(tv), args.n_points), -np.inf, np.float32)
            for v0 in range(0, n_views, 27):
                blk = per_view[:, v0:v0 + 27]            # (Nq, b, D)
                m = ~np.isnan(blk[:, :, 0])
                if not m.any():
                    continue
                qi, vi = np.where(m)
                pv = proj(blk[qi, vi])                   # (n, 64)
                s = tv @ pv.T                            # (Nt, n)
                np.maximum.at(sim_best.T, qi, s.T)
            top1 = sim_best.argmax(axis=1)
            best_d = np.full(len(tv), np.inf)
            for g in gts:
                Rg, tg = g["R"], g["t"].reshape(3, 1) / 1000.0
                for sy in syms:
                    Rs = np.asarray(sy["R"], float)
                    ts = np.asarray(sy["t"], float).reshape(3) / 1000.0
                    posed = (pts_m @ Rs.T + ts) @ Rg.T + tg.reshape(3)
                    d = np.linalg.norm(posed[top1] - tgt.pts, axis=1)
                    best_d = np.minimum(best_d, d)
            r1_max = float((best_d < 0.03 * diam_m).mean())

            # retrieval: crop's pooled patch feature vs each render's pool
            crop64 = tv.mean(0, keepdims=True)
            crop64 = crop64 / np.maximum(np.linalg.norm(crop64), 1e-12)
            vsel = int((view_pool_p @ crop64.T).argmax())
            qv = per_view[:, vsel]
            live = ~np.isnan(qv[:, 0])
            if live.sum() >= 4:
                qv64 = np.zeros((args.n_points, 64), np.float32)
                qv64[live] = proj(qv[live])
                r1_ret, _ = r1_of(qv64)
            else:
                r1_ret = -1.0
            rows.append([sid, im, oid, ci, len(tv),
                         f"{r1_mean:.4f}", f"{r1_max:.4f}", f"{r1_ret:.4f}",
                         vsel])
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(set(todo))} images", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "im_id", "obj_id", "cand", "n_t",
                    "rate1_mean", "rate1_maxview", "rate1_retrieval", "view"])
        w.writerows(rows)
    a = np.array([[float(r[5]), float(r[6]), float(r[7])] for r in rows])
    print(f"rows={len(rows)}  mean rate1: avg={a[:,0].mean():.4f}  "
          f"maxview={a[:,1].mean():.4f}  retrieval={a[:,2].mean():.4f}",
          flush=True)


if __name__ == "__main__":
    main()
