"""FreeZe's Symmetry-Aware Refinement (SAR), reimplemented and measured.

The seven-set analysis left a gap that is entirely orientational: an oracle
picking among rotational alternates of the champion gains +7.0pt AR_MSSD on
YCB-V and +2.1pt on LM-O with zero breakage, while every score already in the
pipeline extracts at most +0.9pt of it. Every one of those scores compares a
QUERY point cloud with a TARGET point cloud in the FUSED (GeDi|DINOv2) feature
space attached to 3D points. FreeZe resolves the same ambiguity somewhere else
entirely -- in image space, between two pictures.

FreeZe v1 (arXiv 2312.00947, Sec. 3.6) defines it as:

  Symmetry estimation  sample SO(3) uniformly; keep the rotations Rs that
                       minimise Chamfer(Rs P_Q, P_Q). Geometry only -- the
                       texture is deliberately ignored here.
  Symmetry selection   render the textured model at each pose Tf o Rs, crop
                       every render with the Sec. 3.3 detection bounding box,
                       encode with Phi (DINOv2), and take the candidate whose
                       PATCH-LEVEL features have the highest mean cosine
                       similarity against V_T, the patch features of the input
                       crop. The paper is explicit that patch tokens, not the
                       class token, carry the pose information, and that both
                       sides are compared "as they are" -- no mapping to
                       points, which is exactly the step our failed criteria
                       all went through.

Its own ablation (Tab. 3) reports SAR as worth +0.4 AR on YCB-V, +0.6 on HB,
+2.6 on IC-BIN and exactly 0.0 on LM-O, T-LESS, TUD-L and ITODD. So the target
here is small and dataset-dependent by construction; LM-O returning nothing is
the published behaviour, not a bug.

Ground truth is never read. Inputs are the candidate poses (already dumped by
flip_rescore_ab.py), the scene RGB/depth, the camera matrix, the detection
masks and the textured CAD model. Scores go out per (target, variant) and are
ranked offline by bop_analyze_flip_rescore.py.

Emitted alongside the faithful score, so that a null result can be attributed
rather than guessed at:

  sar          the paper's rule -- mean cosine over ALL patches, render on
               black, detection bbox.
  sar_comp     render composited into the real image instead of onto black,
               which removes the constant background mismatch that dilutes
               every candidate equally.
  sar_fg       averaged over rendered-foreground patches only.
  cls          class-token cosine. The paper says this is the weaker signal;
               if it wins here, our patch features are the problem.
  ncc, rgb_l1  plain photometry. A control for "is the signal semantic?"
  depth_inl    depth agreement alone. A 180 deg flip of a near-symmetric
               object barely moves the depth map, so this should be blind.
               If it scores, the gain is not coming from appearance.
  chamfer      Chamfer(Rs P_Q, P_Q) / diameter for this variant's rotation,
               letting the paper's symmetry gate be applied offline.

Usage (pod, fresh clone):
  python scripts/sar_render_compare.py --bop /workspace/bop_data/ycbv \
      --dataset ycbv --cand-csv flip_ab3_ycbv.csv \
      --detections .../fastSAM_pbr_ycbv.json --champ-csv .../ycbv_cands.csv \
      --out sar_ycbv.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict, OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch

from popoe.datasets.bop import bop_layout

IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CROP = 224
PATCH = 14
NP_ = CROP // PATCH          # 16x16 patch grid, as in the paper's footnote 3


# --------------------------------------------------------------------------
# mesh assets
# --------------------------------------------------------------------------

def load_asset(mesh_path, device, tex_max=2048, n_sample=2048, seed=0):
    """Load a CAD model with whatever colour information it actually carries.

    `kind` names the colour source that WILL be used: 'texture' (UV atlas),
    'vertex' (per-vertex RGB) or 'lambert' (nothing in the file -- shading
    only). It is reported rather than silently chosen, because a Lambertian
    render of an untextured mesh cannot carry the appearance evidence SAR
    depends on, and a run that quietly degraded to it would read as a negative
    result about SAR instead of a missing asset.
    """
    import trimesh
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    V = np.asarray(mesh.vertices, dtype=np.float32)
    F = np.asarray(mesh.faces, dtype=np.int32)

    norms = np.zeros((len(V), 3), dtype=np.float64)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    for i in range(3):
        np.add.at(norms, F[:, i], fn)
    norms /= (np.linalg.norm(norms, axis=1, keepdims=True) + 1e-8)

    rs = np.random.RandomState(seed)
    try:
        surf, _ = trimesh.sample.sample_surface(mesh, n_sample, seed=seed)
    except TypeError:
        surf, _ = trimesh.sample.sample_surface(mesh, n_sample)
    surf = np.asarray(surf, np.float32)
    surf = surf - surf.mean(0)

    a = dict(V=torch.from_numpy(V).to(device),
             F=torch.from_numpy(F).to(device).contiguous(),
             N=torch.from_numpy(norms.astype(np.float32)).to(device),
             V_np=V, surf=surf,
             diameter=float(np.linalg.norm(V.max(0) - V.min(0))))

    vis = getattr(mesh, 'visual', None)
    uv = getattr(vis, 'uv', None)
    mat = getattr(vis, 'material', None)
    img = getattr(mat, 'image', None) if mat is not None else None
    if uv is not None and img is not None and len(uv) == len(V):
        tex = np.asarray(img, dtype=np.uint8)
        if tex.ndim == 2:
            tex = np.stack([tex] * 3, -1)
        tex = tex[..., :3]
        if max(tex.shape[:2]) > tex_max:
            tex = cv2.resize(tex, (tex_max, tex_max), interpolation=cv2.INTER_AREA)
        tex = tex.astype(np.float32) / 255.0
        uv = np.asarray(uv, dtype=np.float32).copy()
        # dr.texture puts v=0 at the FIRST texture row; PLY/OBJ atlases put it
        # at the bottom. Same flip the query-view renderer needs (2026-07-06).
        uv[:, 1] = 1.0 - uv[:, 1]
        a['uv'] = torch.from_numpy(uv).to(device)
        a['tex'] = torch.from_numpy(np.ascontiguousarray(tex)).to(device)
        a['kind'] = 'texture'
        return a

    vc = None
    if vis is not None:
        try:
            vc = np.asarray(vis.vertex_colors, dtype=np.float32)[:, :3] / 255.0
        except Exception:
            vc = None
    if vc is not None and len(vc) == len(V) and float(vc.std()) > 1e-3:
        a['vcol'] = torch.from_numpy(np.ascontiguousarray(vc.astype(np.float32))).to(device)
        a['kind'] = 'vertex'
        return a

    a['kind'] = 'lambert'
    return a


def chamfer_norm(surf, R, diameter):
    """Symmetric Chamfer between R*P and P, in units of the object diameter.

    This is the paper's symmetry criterion (Sec. 3.6). Small value = the
    rotation maps the shape onto itself, i.e. it is a genuine geometric
    symmetry and a legitimate SAR candidate.
    """
    from sklearn.neighbors import KDTree
    P = surf
    Q = P @ np.asarray(R, np.float32).T
    d1, _ = KDTree(P).query(Q, k=1)
    d2, _ = KDTree(Q).query(P, k=1)
    return float((d1.mean() + d2.mean()) / 2.0 / max(diameter, 1e-6))


# --------------------------------------------------------------------------
# rasteriser
# --------------------------------------------------------------------------

class PoseRenderer:
    """nvdiffrast rasteriser driven by a real camera matrix and an object pose.

    Everything is millimetres, matching BOP and the `t` column of the dump.

    `y_sign` records which screen row nvdiffrast's NDC +1 lands on. That is a
    property of the build, not something to assume: `calibrate()` settles it
    against a plain numpy pinhole projection of the same vertices, because
    getting it backwards renders every object upside down and would silently
    turn this into an experiment about vertical mirror symmetry.
    """

    def __init__(self, device='cuda'):
        import nvdiffrast.torch as dr
        self.dr = dr
        self.ctx = dr.RasterizeCudaContext(device=device)
        self.device = device
        self.y_sign = 1.0

    def _clip(self, V_cam, K, H, W):
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        z = V_cam[:, 2]
        near, far = 1.0, 1.0e7
        x_clip = (2.0 * fx * V_cam[:, 0] + (2.0 * cx - W) * z) / W
        y_clip = self.y_sign * (2.0 * fy * V_cam[:, 1] + (2.0 * cy - H) * z) / H
        z_clip = ((far + near) / (far - near)) * z - (2 * far * near) / (far - near)
        return torch.stack([x_clip, y_clip, z_clip, z], dim=1).contiguous()

    def render(self, asset, R, t, K, H, W, ambient=1.0):
        """Return (rgb float32 HxWx3 in [0,1] on black, mask bool, depth mm)."""
        Rt = torch.as_tensor(np.ascontiguousarray(np.asarray(R, np.float32).T),
                             device=self.device)
        tt = torch.as_tensor(np.asarray(t, np.float32).reshape(1, 3),
                             device=self.device)
        V_cam = asset['V'] @ Rt + tt
        pos = self._clip(V_cam, K, H, W).unsqueeze(0)
        tri = asset['F']
        rast, _ = self.dr.rasterize(self.ctx, pos, tri, resolution=[H, W])
        hit = rast[0, :, :, 3] > 0

        if asset['kind'] == 'texture':
            uv_i, _ = self.dr.interpolate(asset['uv'].unsqueeze(0), rast, tri)
            col = self.dr.texture(asset['tex'].unsqueeze(0), uv_i,
                                  filter_mode='linear')[0]
        elif asset['kind'] == 'vertex':
            col, _ = self.dr.interpolate(asset['vcol'].unsqueeze(0), rast, tri)
            col = col[0]
        else:
            col = torch.full((H, W, 3), 0.70, device=self.device)

        if ambient < 1.0:
            n_cam = (asset['N'] @ Rt).contiguous()
            n_i, _ = self.dr.interpolate(n_cam.unsqueeze(0), rast, tri)
            n_i = torch.nn.functional.normalize(n_i[0], dim=-1)
            # headlight: camera looks down +z, so a surface facing it has n_z<0
            ndl = torch.clamp(-n_i[..., 2], 0.0, 1.0)
            col = col * (ambient + (1.0 - ambient) * ndl).unsqueeze(-1)

        zc, _ = self.dr.interpolate(V_cam[:, 2:3].contiguous().unsqueeze(0),
                                    rast, tri)
        depth = torch.where(hit, zc[0, :, :, 0], torch.zeros_like(zc[0, :, :, 0]))
        rgb = torch.where(hit.unsqueeze(-1), col.clamp(0, 1),
                          torch.zeros_like(col))
        return rgb.cpu().numpy(), hit.cpu().numpy(), depth.cpu().numpy()

    def calibrate(self, asset, R, t, K, H, W):
        """Pin y_sign against a numpy pinhole projection. Returns (sign, iou)."""
        V = asset['V_np']
        Vc = V @ np.asarray(R, np.float32).T + np.asarray(t, np.float32)
        ok = Vc[:, 2] > 1e-3
        u = (Vc[ok, 0] / Vc[ok, 2]) * K[0, 0] + K[0, 2]
        v = (Vc[ok, 1] / Vc[ok, 2]) * K[1, 1] + K[1, 2]
        ref = np.zeros((H, W), np.uint8)
        ref[np.clip(v.astype(int), 0, H - 1), np.clip(u.astype(int), 0, W - 1)] = 1
        ref = cv2.dilate(ref, np.ones((5, 5), np.uint8)) > 0
        best = (1.0, -1.0)
        for s in (1.0, -1.0):
            self.y_sign = s
            _, m, _ = self.render(asset, R, t, K, H, W)
            iou = (m & ref).sum() / max((m | ref).sum(), 1)
            if iou > best[1]:
                best = (s, iou)
        self.y_sign = best[0]
        return best


# --------------------------------------------------------------------------
# crop helpers
# --------------------------------------------------------------------------

def boxes_of(mask, pad=1.25, min_side=24):
    """Both readings of "the bounding box defined in Sec. 3.3".

    'ti' is the literal one -- the minimum box containing the mask, resized
    (and therefore stretched) to 224. 'sq' pads it to a square first, which
    preserves aspect ratio but, for a tall thin object like the mustard
    bottle, leaves it covering only about a quarter of the patch grid; the
    other three quarters are background that mismatches identically for every
    candidate and dilutes the mean cosine. Which reading the authors used is
    not stated, so both are measured.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
    hh = max((y1 - y0) * pad, min_side) / 2.0
    hw = max((x1 - x0) * pad, min_side) / 2.0
    side = max(hh, hw)
    return {
        "sq": (int(round(cx - side)), int(round(cy - side)),
               int(round(cx + side)), int(round(cy + side))),
        "ti": (int(round(cx - hw)), int(round(cy - hh)),
               int(round(cx + hw)), int(round(cy + hh))),
    }


def crop_resize(img, box, interp=cv2.INTER_LINEAR):
    """Crop with zero padding outside the image, then resize to CROP."""
    x0, y0, x1, y1 = box
    H, W = img.shape[:2]
    out = np.zeros((y1 - y0, x1 - x0) + img.shape[2:], dtype=img.dtype)
    sy0, sy1, sx0, sx1 = max(y0, 0), min(y1, H), max(x0, 0), min(x1, W)
    if sy1 > sy0 and sx1 > sx0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return cv2.resize(out, (CROP, CROP), interpolation=interp)


def patch_pool(m224):
    """(224,224) float coverage -> (16,16) mean coverage per DINO patch."""
    return m224.reshape(NP_, PATCH, NP_, PATCH).mean(axis=(1, 3))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bop", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cand-csv", required=True,
                    help="flip_ab3_<ds>.csv: the candidate poses to score")
    ap.add_argument("--out", required=True)
    ap.add_argument("--detections", default="",
                    help="CNOS/FastSAM json; with --champ-csv gives the "
                         "paper's Sec. 3.3 detection bbox")
    ap.add_argument("--champ-csv", default="",
                    help="<ds>_cands.csv, to recover which detection won")
    ap.add_argument("--objs", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ambient", type=float, default=1.0,
                    help="1.0 = flat albedo (paper renders the textured model);"
                         " <1 adds a headlight Lambert term")
    ap.add_argument("--occ-tol", type=float, default=20.0,
                    help="mm; observed depth this far IN FRONT of the render "
                         "marks the pixel occluded")
    ap.add_argument("--depth-tol", type=float, default=10.0,
                    help="mm; agreement band for the depth-only control")
    ap.add_argument("--pad", type=float, default=1.25)
    ap.add_argument("--min-cov", type=float, default=0.5,
                    help="a patch is foreground if this much of it is rendered")
    ap.add_argument("--tex-max", type=int, default=2048)
    ap.add_argument("--debug-dir", default="")
    ap.add_argument("--debug-obj", type=int, default=0)
    ap.add_argument("--debug-n", type=int, default=16)
    ap.add_argument("--batch", type=int, default=13)
    args = ap.parse_args()

    bop = Path(args.bop)
    layout = bop_layout(args.dataset)
    device = 'cuda'

    rows = defaultdict(list)
    with open(args.cand_csv) as f:
        for r in csv.DictReader(f):
            k = (int(r["scene_id"]), int(r["im_id"]), int(r["obj_id"]))
            rows[k].append((r["variant"],
                            np.array([float(x) for x in r["R"].split()]).reshape(3, 3),
                            np.array([float(x) for x in r["t"].split()], float)))
    obj_filter = {int(x) for x in args.objs.split(",") if x.strip()}
    keys = sorted(k for k in rows if not obj_filter or k[2] in obj_filter)
    if args.limit:
        keys = keys[:args.limit]
    print(f"{len(keys)} targets, {sum(len(rows[k]) for k in keys)} candidates",
          flush=True)

    # ---- detection bbox (paper Sec. 3.3). Optional: without it the crop is
    # taken from the union of the candidate renders, which is still identical
    # across variants (the property that matters for comparability) but is
    # derived from the hypothesis rather than from the observation.
    segmentor = champs = None
    if args.detections and args.champ_csv:
        from popoe.freeze.recipes import YCBV_MERGE_LABELS, best_segmentor
        merge = dict(YCBV_MERGE_LABELS) if args.dataset == "ycbv" else {}
        segmentor = best_segmentor(args.detections, topk=2, merge_labels=merge)
        champs = {}
        for r in csv.DictReader(open(args.champ_csv)):
            k = (int(r["scene_id"]), int(r["im_id"]), int(r["obj_id"]))
            s = float(r["score"])
            if k not in champs or s > champs[k]["score"]:
                champs[k] = dict(score=s, cand=int(r["cand"]))
        print(f"detection bbox mode: {len(champs)} champion detections",
              flush=True)
    else:
        print("bbox mode: union of candidate renders (no --detections)",
              flush=True)

    print("loading DINOv2 ...", flush=True)
    from popoe.freeze.feature_extractor import load_dinov2, _dino_layer
    dino = load_dinov2(device)
    layer = _dino_layer(dino)
    print(f"  dinov2_vitg14_reg, {dino.n_blocks} blocks, patch layer {layer}",
          flush=True)

    renderer = PoseRenderer(device)
    assets = OrderedDict()

    def asset_of(obj_id):
        if obj_id not in assets:
            p = str(bop / layout["models_dir"] / f"obj_{obj_id:06d}.ply")
            a = load_asset(p, device, tex_max=args.tex_max)
            print(f"  obj {obj_id}: {len(a['V_np'])} verts, colour={a['kind']}, "
                  f"diam={a['diameter']:.1f}mm", flush=True)
            assets[obj_id] = a
        return assets[obj_id]

    cham_cache = {}

    def chamfer_of(obj_id, R_rel):
        k = (obj_id, np.round(R_rel, 3).tobytes())
        if k not in cham_cache:
            a = asset_of(obj_id)
            cham_cache[k] = chamfer_norm(a['surf'], R_rel, a['diameter'])
        return cham_cache[k]

    by_img = defaultdict(list)
    for k in keys:
        by_img[(k[0], k[1])].append(k[2])

    dbg_dir = Path(args.debug_dir) if args.debug_dir else None
    if dbg_dir:
        dbg_dir.mkdir(parents=True, exist_ok=True)
    n_dbg = 0

    fout = open(args.out, "w", newline="")
    wr = csv.writer(fout)
    # 'ti' = the paper's literal minimum bounding box, stretched to 224.
    # 'sq' = the same box padded to a square first. `sar_ti` is the faithful
    # reading of Sec. 3.6; everything else is an ablation around it.
    wr.writerow(["scene_id", "im_id", "obj_id", "variant",
                 "sar_ti", "sar_ti_comp", "sar_ti_fg", "cls_ti",
                 "sar_sq", "sar_sq_comp", "sar_sq_fg", "cls_sq",
                 "ncc", "rgb_l1", "depth_inl", "chamfer",
                 "n_mask", "n_patch", "n_occ", "bbox_src"])

    calibrated = False
    n_done = n_skip = 0
    for (scene_id, im_id), objs in sorted(by_img.items()):
        sdir = bop / layout["split"] / f"{scene_id:06d}"
        try:
            cam = json.load(open(sdir / "scene_camera.json"))[str(im_id)]
        except Exception:
            continue
        K = np.array(cam["cam_K"], float).reshape(3, 3)
        bgr = cv2.imread(str(sdir / layout["img_dir"] / f"{im_id:06d}{layout['img_ext']}"),
                         cv2.IMREAD_UNCHANGED)
        draw = cv2.imread(str(sdir / "depth" / f"{im_id:06d}{layout['depth_ext']}"),
                          cv2.IMREAD_UNCHANGED)
        if bgr is None:
            continue
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(bgr[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        H, W = rgb.shape[:2]
        obs_d = (draw.astype(np.float32) * float(cam["depth_scale"])
                 if draw is not None else np.zeros((H, W), np.float32))
        grey_real = rgb.mean(-1)

        scene = None
        if segmentor is not None:
            from popoe.interfaces import Scene
            depth_m = obs_d / 1000.0
            scene = Scene(rgb=(rgb * 255).astype(np.uint8), depth=depth_m, K=K,
                          scene_id=scene_id, im_id=im_id)

        for obj_id in sorted(objs):
            key = (scene_id, im_id, obj_id)
            cands = rows[key]
            a = asset_of(obj_id)

            if not calibrated:
                s, iou = renderer.calibrate(a, cands[0][1], cands[0][2], K, H, W)
                print(f"[calib] nvdiffrast y_sign={s:+.0f} (IoU {iou:.3f} vs "
                      f"numpy pinhole projection)", flush=True)
                if iou < 0.5:
                    raise SystemExit("rasteriser disagrees with the pinhole "
                                     "projection under BOTH y conventions -- "
                                     "renders would be geometrically wrong")
                calibrated = True

            R_ = [renderer.render(a, Rv, tv, K, H, W, args.ambient)
                  for _, Rv, tv in cands]
            masks = [m for _, m, _ in R_]

            boxes, bbox_src = None, "render"
            if segmentor is not None and key in champs:
                try:
                    from popoe.interfaces import ObjectModel
                    obj = ObjectModel(obj_id=obj_id, mesh_path="", diameter=0.0)
                    dets = segmentor.segment(scene, obj)
                    ci = champs[key]["cand"]
                    if ci < len(dets):
                        boxes = boxes_of(dets[ci].mask, args.pad)
                        bbox_src = "det"
                except Exception as e:
                    if n_done == 0:
                        print(f"[warn] segmentor failed ({e}); using render bbox",
                              flush=True)
            if boxes is None:
                union = np.zeros((H, W), bool)
                for m in masks:
                    union |= m
                boxes = boxes_of(union, args.pad)
            if boxes is None:
                n_skip += 1
                for name, _, _ in cands:
                    wr.writerow([scene_id, im_id, obj_id, name] + [""] * 15)
                continue

            R_champ = next(Rv for nm, Rv, _ in cands if nm == "champion")
            nc = len(cands)
            per_box = {}
            extras = None
            for tag, box in boxes.items():
                real_c = crop_resize(rgb, box)
                obs_dc = crop_resize(obs_d, box, cv2.INTER_NEAREST)
                grey_c = crop_resize(grey_real, box)

                black_imgs, comp_imgs, pmasks = [], [], []
                ex = []
                for (name, Rv, _), (rr, mm, dd) in zip(cands, R_):
                    rc = crop_resize(rr, box)
                    mc = crop_resize(mm.astype(np.float32), box)
                    dc = crop_resize(dd, box, cv2.INTER_NEAREST)
                    hard = mc > 0.5
                    occ = hard & (obs_dc > 1.0) & (obs_dc < dc - args.occ_tol)

                    comp = real_c.copy()
                    comp[hard] = rc[hard]

                    cov = patch_pool(mc)
                    occ_p = patch_pool(occ.astype(np.float32))
                    pmasks.append((cov >= args.min_cov) & (occ_p < 0.5))

                    use = hard & (~occ)
                    if use.sum() >= 16:
                        x = grey_c[use] - grey_c[use].mean()
                        y = rc[..., :3].mean(-1)[use]
                        y = y - y.mean()
                        den = math.sqrt(float((x * x).sum()) *
                                        float((y * y).sum())) + 1e-8
                        ncc = float((x * y).sum() / den)
                        l1 = float(np.abs(rc[use] - real_c[use]).mean())
                    else:
                        ncc, l1 = float('nan'), float('nan')
                    dv = hard & (obs_dc > 1.0)
                    dinl = (float((np.abs(obs_dc[dv] - dc[dv]) < args.depth_tol).mean())
                            if dv.sum() >= 16 else float('nan'))
                    cham = chamfer_of(obj_id, R_champ.T @ Rv)

                    black_imgs.append(rc)               # paper: model on black
                    comp_imgs.append(comp)
                    ex.append((ncc, l1, dinl, cham, int(mm.sum()), int(occ.sum())))
                if extras is None:                      # box-independent columns
                    extras = ex

                arr = np.stack([real_c] + black_imgs + comp_imgs).astype(np.float32)
                arr = (arr - IMNET_MEAN) / IMNET_STD
                ten = torch.from_numpy(arr.transpose(0, 3, 1, 2)).to(device)
                with torch.no_grad():
                    pts, cts = [], []
                    for i in range(0, len(ten), args.batch):
                        o = dino.get_intermediate_layers(
                            ten[i:i + args.batch], n=[layer],
                            return_class_token=True)[0]
                        pts.append(o[0].float())
                        cts.append(o[1].float())
                    pt = torch.nn.functional.normalize(torch.cat(pts), dim=-1)
                    ct = torch.nn.functional.normalize(torch.cat(cts), dim=-1)
                per_box[tag] = ((pt[1:] * pt[0].unsqueeze(0)).sum(-1).cpu().numpy(),
                                (ct[1:] * ct[0].unsqueeze(0)).sum(-1).cpu().numpy(),
                                pmasks)
                if tag == "ti" and dbg_dir and n_dbg < args.debug_n and \
                        (not args.debug_obj or obj_id == args.debug_obj):
                    top = np.concatenate([real_c] + black_imgs, axis=1)
                    bot = np.concatenate([real_c] + comp_imgs, axis=1)
                    out = (np.concatenate([top, bot], axis=0) * 255).astype(np.uint8)
                    cv2.imwrite(str(dbg_dir /
                                    f"{scene_id:06d}_{im_id:06d}_{obj_id}.png"),
                                cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
                    n_dbg += 1

            for j, (name, _, _) in enumerate(cands):
                out = [scene_id, im_id, obj_id, name]
                npx = 0
                for tag in ("ti", "sq"):
                    cos_p, cos_c, pmasks = per_box[tag]
                    pm = pmasks[j].reshape(-1)
                    n = int(pm.sum())
                    if tag == "ti":
                        npx = n
                    out += [f"{float(cos_p[j].mean()):.6f}",          # paper rule
                            f"{float(cos_p[nc + j].mean()):.6f}",     # composited
                            (f"{float(cos_p[j][pm].mean()):.6f}" if n else ""),
                            f"{float(cos_c[j]):.6f}"]
                ncc, l1, dinl, cham, nmask, nocc = extras[j]
                out += [f"{ncc:.6f}", f"{l1:.6f}", f"{dinl:.6f}", f"{cham:.6f}",
                        nmask, npx, nocc, bbox_src]
                wr.writerow(out)

            n_done += 1
            if n_done % 100 == 0:
                print(f"  {n_done} targets", flush=True)
                fout.flush()
    fout.close()
    print(f"done: {n_done} targets, {n_skip} empty renders -> {args.out}",
          flush=True)
    print("colour sources: " +
          ", ".join(f"{o}:{a['kind']}" for o, a in assets.items()), flush=True)


if __name__ == "__main__":
    main()
