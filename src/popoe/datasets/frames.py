"""Generic RGB-D frame manifests for non-BOP scenes.

The BOP path already has a dataset loader. Real captures are looser: one frame
usually lives as RGB/depth files plus intrinsics and a detections JSON produced
by whichever front-end ran (CNOS-style, SAM prompt, open-vocab, ...). This
module turns that file manifest into the same `Scene` object the pose pipeline
already consumes.

Depth convention: raw_depth * depth_scale -> metres. Detections remain separate
2D masks; they do not carry depth.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np

from popoe.interfaces import FrameManifest, Scene


def _resolve_path(path: Optional[Union[str, os.PathLike]],
                  base_dir: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    p = os.fspath(path)
    if os.path.isabs(p) or base_dir is None:
        return p
    return os.path.normpath(os.path.join(base_dir, p))


def _read_npy_or_image(path: str, *, rgb: bool) -> np.ndarray:
    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        return np.load(path)

    try:
        import cv2
    except ImportError:
        cv2 = None

    if cv2 is not None:
        flag = cv2.IMREAD_COLOR if rgb else cv2.IMREAD_UNCHANGED
        arr = cv2.imread(path, flag)
        if arr is None:
            raise FileNotFoundError(path)
        if rgb:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return arr

    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            f"reading image files needs opencv-python or Pillow; cannot load {path!r}"
        ) from e
    arr = np.asarray(Image.open(path))
    if rgb and arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    return arr


def load_frame_manifest(path_or_record, *,
                        base_dir: Optional[str] = None) -> FrameManifest:
    """Load a frame manifest from JSON path or dict.

    Required fields:
        rgb_path, depth_path, K

    Optional fields:
        scene_id, image_id/im_id, depth_scale, detections_path

    Relative paths are resolved against the manifest JSON's directory, or
    `base_dir` when passing a dict directly.
    """
    if isinstance(path_or_record, (str, os.PathLike)):
        path = os.fspath(path_or_record)
        with open(path) as f:
            rec = json.load(f)
        base = os.path.dirname(os.path.abspath(path))
    else:
        rec = dict(path_or_record)
        base = base_dir

    im_id = rec.get("im_id", rec.get("image_id", -1))
    K = rec.get("K", rec.get("cam_K"))
    if K is None:
        raise KeyError("frame manifest needs K or cam_K")
    return FrameManifest(
        rgb_path=_resolve_path(rec["rgb_path"], base),
        depth_path=_resolve_path(rec["depth_path"], base),
        K=K,
        depth_scale=rec.get("depth_scale", 1.0),
        scene_id=rec.get("scene_id", -1),
        im_id=im_id,
        detections_path=_resolve_path(rec.get("detections_path"), base),
        meta={k: v for k, v in rec.items()
              if k not in {"rgb_path", "depth_path", "K", "cam_K",
                           "depth_scale", "scene_id", "image_id", "im_id",
                           "detections_path"}},
    )


def load_scene_from_manifest(path_or_manifest) -> Scene:
    """Load RGB/depth arrays and return a `Scene` with depth in metres."""
    manifest = (path_or_manifest if isinstance(path_or_manifest, FrameManifest)
                else load_frame_manifest(path_or_manifest))
    rgb = _read_npy_or_image(manifest.rgb_path, rgb=True)
    depth_raw = _read_npy_or_image(manifest.depth_path, rgb=False)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must be HxWx3, got {rgb.shape}")
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[:, :, 0]
    if depth_raw.ndim != 2:
        raise ValueError(f"depth must be HxW, got {depth_raw.shape}")
    if rgb.shape[:2] != depth_raw.shape:
        raise ValueError(
            f"rgb/depth shape mismatch: {rgb.shape[:2]} vs {depth_raw.shape}"
        )
    depth = depth_raw.astype(np.float32) * manifest.depth_scale
    return Scene(rgb=rgb.astype(np.uint8, copy=False),
                 depth=depth.astype(np.float32, copy=False),
                 K=manifest.K,
                 scene_id=manifest.scene_id,
                 im_id=manifest.im_id)


__all__ = ["FrameManifest", "load_frame_manifest", "load_scene_from_manifest"]
