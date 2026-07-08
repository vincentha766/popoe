"""Minimal BOP dataset helpers (test split) — find instances and load RGB-D + GT.

Standard BOP layout under `bop_root`:
    test/000048/rgb/000001.png  depth/000001.png  mask_visib/000001_000002.png
                scene_camera.json  scene_gt.json
    models/obj_000005.ply
Depth is returned in METRES (raw uint16 * depth_scale / 1000).
"""
import os
import glob
import json
import numpy as np
import cv2


def find_instances(bop_root, obj_id, n=5):
    """Return up to `n` (scene_id, im_id, gt_idx) triples for `obj_id`."""
    out = []
    for p in sorted(glob.glob(f"{bop_root}/test/*/scene_gt.json")):
        scene_id = int(os.path.basename(os.path.dirname(p)))
        gt = json.load(open(p))
        for im_str, ents in gt.items():
            for gi, e in enumerate(ents):
                if e["obj_id"] == obj_id:
                    m = (f"{bop_root}/test/{scene_id:06d}/mask_visib/"
                         f"{int(im_str):06d}_{gi:06d}.png")
                    if os.path.exists(m):
                        out.append((scene_id, int(im_str), gi))
                        if len(out) >= n:
                            return out
    return out


def load_inputs(bop_root, scene_id, im_id, gt_idx):
    """Return (rgb uint8 HxWx3, depth float32 metres, mask bool, K 3x3, intr dict)."""
    sd = f"{bop_root}/test/{scene_id:06d}"
    cam = json.load(open(f"{sd}/scene_camera.json"))[str(im_id)]
    K = np.array(cam["cam_K"], np.float64).reshape(3, 3)
    rgb = cv2.cvtColor(cv2.imread(f"{sd}/rgb/{im_id:06d}.png"), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(f"{sd}/depth/{im_id:06d}.png", cv2.IMREAD_UNCHANGED).astype(
        np.float32) * cam["depth_scale"] / 1000.0
    mask = cv2.imread(f"{sd}/mask_visib/{im_id:06d}_{gt_idx:06d}.png",
                      cv2.IMREAD_UNCHANGED) > 0
    intr = {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2]}
    return rgb, depth, mask, K, intr


def load_gt(bop_root, scene_id, im_id, gt_idx):
    """Return (R_m2c 3x3, t_m2c mm) ground-truth pose."""
    gt = json.load(open(f"{bop_root}/test/{scene_id:06d}/scene_gt.json"))[str(im_id)][gt_idx]
    return (np.array(gt["cam_R_m2c"], np.float64).reshape(3, 3),
            np.array(gt["cam_t_m2c"], np.float64))
