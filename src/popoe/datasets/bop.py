"""Minimal BOP dataset helpers — find instances and load BOP RGB-D frames.

Common BOP layout under `bop_root`:
    test/000048/rgb/000001.png  depth/000001.png  mask_visib/000001_000002.png
                scene_camera.json  scene_gt.json
    models/obj_000005.ply
Not every dataset follows it — split dir, image modality/extension and models
dir vary per dataset; the deviations are tabulated in BOP_LAYOUTS below.
Depth is returned in METRES (raw uint16 * depth_scale / 1000).
"""
import glob
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from popoe.datasets.frames import load_scene_from_manifest
from popoe.interfaces import FrameManifest, Scene


def _read_json(path):
    with open(path) as f:
        return json.load(f)


# Per-dataset directory layout for the BOP-Classic-Core seven. Values follow
# bop_toolkit's dataset_params (its defaults: tless split_type='primesense',
# model_type='cad'; itodd images are gray/*.tif with depth/*.tif). Everything
# outside this table is NOT guessed — an unknown dataset must be named
# explicitly by the caller, because the failure mode of a wrong guess is not a
# crash: a missing image dir used to yield a clean-looking all-zero CSV.
BOP_LAYOUTS = {
    "lmo":   {"split": "test", "img_dir": "rgb", "img_ext": ".png",
              "depth_ext": ".png", "models_dir": "models"},
    "tless": {"split": "test_primesense", "img_dir": "rgb", "img_ext": ".png",
              "depth_ext": ".png", "models_dir": "models_cad"},
    "tudl":  {"split": "test", "img_dir": "rgb", "img_ext": ".png",
              "depth_ext": ".png", "models_dir": "models"},
    "icbin": {"split": "test", "img_dir": "rgb", "img_ext": ".png",
              "depth_ext": ".png", "models_dir": "models"},
    "itodd": {"split": "test", "img_dir": "gray", "img_ext": ".tif",
              "depth_ext": ".tif", "models_dir": "models"},
    "hb":    {"split": "test_primesense", "img_dir": "rgb", "img_ext": ".png",
              "depth_ext": ".png", "models_dir": "models"},
    "ycbv":  {"split": "test", "img_dir": "rgb", "img_ext": ".png",
              "depth_ext": ".png", "models_dir": "models"},
}


def bop_layout(dataset, split=None, models_dir=None) -> dict:
    """Resolve a dataset's directory layout, with explicit overrides.

    Raises ValueError for a name outside BOP_LAYOUTS — overrides refine a
    known layout (e.g. hb's test_kinect), they do not define a new one, since
    the image dir/ext would still be a guess.
    """
    if dataset not in BOP_LAYOUTS:
        raise ValueError(
            f"unknown BOP dataset {dataset!r}; known: "
            f"{', '.join(sorted(BOP_LAYOUTS))}")
    layout = dict(BOP_LAYOUTS[dataset])
    if split is not None:
        layout["split"] = split
    if models_dir is not None:
        layout["models_dir"] = models_dir
    return layout


def default_targets_path(bop_root, split="test") -> Path:
    """Return the conventional BOP target file path for a split."""

    root = Path(bop_root)
    path = root / f"{split}_targets_bop19.json"
    if path.exists():
        return path
    fallback = root / "test_targets_bop19.json"
    if split.startswith("test_") and fallback.exists():
        return fallback
    return path


def bop_frame_manifest(bop_root, split, scene_id, im_id, img_dir="rgb",
                       img_ext=".png", depth_ext=".png") -> FrameManifest:
    """Build a frame manifest for one BOP image.

    BOP ``scene_camera.json`` stores ``depth_scale`` in millimetres per raw
    depth unit. ``FrameManifest`` expects metres per raw unit, so divide by 1000.
    The image dir/extensions default to the rgb/png layout most sets use; pass
    the values from ``bop_layout`` for the ones that differ (itodd: gray/tif).
    """

    root = str(Path(bop_root).expanduser())
    scene_dir = Path(root) / split / f"{int(scene_id):06d}"
    cameras = _scene_camera(root, split, int(scene_id))
    cam = cameras[str(int(im_id))]
    return FrameManifest(
        rgb_path=str(scene_dir / img_dir / f"{int(im_id):06d}{img_ext}"),
        depth_path=str(scene_dir / "depth" / f"{int(im_id):06d}{depth_ext}"),
        K=cam["cam_K"],
        depth_scale=float(cam.get("depth_scale", 1.0)) / 1000.0,
        scene_id=int(scene_id),
        im_id=int(im_id),
    )


@lru_cache(maxsize=256)
def _scene_camera(bop_root: str, split: str, scene_id: int) -> dict:
    scene_dir = Path(bop_root) / split / f"{int(scene_id):06d}"
    return _read_json(scene_dir / "scene_camera.json")


def load_bop_scene(bop_root, split, scene_id, im_id) -> Scene:
    """Load one BOP RGB-D frame as a ``Scene`` with depth in metres."""

    return load_scene_from_manifest(
        bop_frame_manifest(bop_root, split, scene_id, im_id)
    )


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
    import cv2

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
