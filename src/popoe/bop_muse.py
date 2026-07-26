"""Run the MUSE reimplementation over a BOP target split.

This is the split-wide producer for ``segmentor_muse``. It turns BOP target
images into ``Scene`` objects, runs one registered ``MuseSegmentor`` over each
image, and writes a BOP-format detections JSON with ``source='muse-repro'``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from popoe.datasets.frames import load_scene_from_manifest
from popoe.interfaces import FrameManifest, ObjectModel, Scene
from popoe.segmentor_muse import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_GAMMA,
    DEFAULT_GDINO_MODEL,
    DEFAULT_MIN_PIXELS,
    DEFAULT_PROMPT,
    DEFAULT_TAU,
    MUSE_SOURCE,
    MuseClass,
    build_muse_segmentor,
    muse_records,
    write_muse_detections,
)


@dataclass(frozen=True)
class BOPTargetImage:
    """One BOP image plus the object ids requested for that image."""

    scene_id: int
    im_id: int
    obj_ids: tuple[int, ...]


SceneLoader = Callable[[str | os.PathLike, str, int, int], Scene]
ProgressCallback = Callable[[int, int, BOPTargetImage, int, float, bool], None]


def _read_json(path: str | os.PathLike):
    with open(path) as f:
        return json.load(f)


def _parse_obj_ids(text: str | None) -> set[int]:
    if not text:
        return set()
    return {int(x.strip()) for x in text.split(",") if x.strip()}


def _parse_template_overrides(spec: str | None) -> dict[int, str]:
    """Parse ``'9=/tpl/9,14=/tpl/14'`` into explicit template dirs."""

    out: dict[int, str] = {}
    if not spec:
        return out
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"--classes entry needs 'obj_id=template_dir', got {item!r}")
        oid, path = item.split("=", 1)
        obj_id = int(oid)
        if obj_id in out:
            raise ValueError(f"duplicate template override for obj_id {obj_id}")
        out[obj_id] = path.strip()
    return out


def default_targets_path(bop_root: str | os.PathLike, split: str = "test") -> Path:
    """Return the conventional BOP target file path for a split."""

    return Path(bop_root) / f"{split}_targets_bop19.json"


def load_bop_target_images(
    targets_json: str | os.PathLike,
    *,
    obj_ids: set[int] | Sequence[int] | None = None,
    limit_images: int | None = None,
) -> list[BOPTargetImage]:
    """Load and group BOP targets by image.

    BOP target files repeat rows when an object has ``inst_count > 1``. For
    segmentation production the image is processed once; the grouped ``obj_ids``
    are used for optional output filtering and smoke subsets.
    """

    keep = {int(x) for x in obj_ids} if obj_ids else None
    grouped: dict[tuple[int, int], set[int]] = {}
    for rec in _read_json(targets_json):
        obj_id = int(rec["obj_id"])
        if keep is not None and obj_id not in keep:
            continue
        key = (int(rec["scene_id"]), int(rec["im_id"]))
        grouped.setdefault(key, set()).add(obj_id)

    out = [
        BOPTargetImage(scene_id=sid, im_id=iid, obj_ids=tuple(sorted(ids)))
        for (sid, iid), ids in sorted(grouped.items())
    ]
    if limit_images is not None and int(limit_images) > 0:
        out = out[: int(limit_images)]
    return out


def load_models_info(models_info: str | os.PathLike | Mapping) -> dict:
    """Load BOP ``models_info.json`` or normalise an already-loaded mapping."""

    raw = _read_json(models_info) if not isinstance(models_info, Mapping) else models_info
    return {str(k): dict(v) for k, v in raw.items()}


def _template_dir(
    obj_id: int,
    *,
    template_root: str | os.PathLike | None,
    template_pattern: str,
    template_overrides: Mapping[int, str] | None,
) -> str:
    overrides = {int(k): str(v) for k, v in (template_overrides or {}).items()}
    if obj_id in overrides:
        path = Path(overrides[obj_id])
    elif template_root:
        path = Path(template_root) / template_pattern.format(obj_id=int(obj_id))
    else:
        raise ValueError(
            f"no template directory for obj_id {obj_id}; pass --template-root "
            "or an explicit --classes obj_id=template_dir entry"
        )
    if not path.exists():
        raise FileNotFoundError(f"template directory not found for obj_id {obj_id}: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"template path is not a directory for obj_id {obj_id}: {path}")
    return str(path)


def build_muse_classes(
    obj_ids: Sequence[int],
    *,
    models_info: str | os.PathLike | Mapping,
    models_dir: str | os.PathLike | None = None,
    template_root: str | os.PathLike | None = None,
    template_pattern: str = "obj_{obj_id:06d}",
    template_overrides: Mapping[int, str] | None = None,
) -> list[MuseClass]:
    """Build ``MuseClass`` registrations from BOP metadata and templates."""

    info = load_models_info(models_info)
    out: list[MuseClass] = []
    for obj_id in sorted({int(x) for x in obj_ids}):
        key = str(obj_id)
        if key not in info:
            raise KeyError(f"obj_id {obj_id} not in models_info")
        mesh_path = (
            str(Path(models_dir) / f"obj_{obj_id:06d}.ply")
            if models_dir is not None else ""
        )
        out.append(MuseClass(
            obj=ObjectModel(
                obj_id=obj_id,
                mesh_path=mesh_path,
                diameter=float(info[key]["diameter"]) / 1000.0,
            ),
            template_dir=_template_dir(
                obj_id,
                template_root=template_root,
                template_pattern=template_pattern,
                template_overrides=template_overrides,
            ),
        ))
    return out


def bop_frame_manifest(
    bop_root: str | os.PathLike,
    split: str,
    scene_id: int,
    im_id: int,
) -> FrameManifest:
    """Build a frame manifest for one BOP image.

    BOP ``scene_camera.json`` stores ``depth_scale`` in millimetres per raw
    depth unit. ``FrameManifest`` expects metres per raw unit, so divide by 1000.
    """

    scene_dir = Path(bop_root) / split / f"{int(scene_id):06d}"
    cameras = _read_json(scene_dir / "scene_camera.json")
    cam = cameras[str(int(im_id))]
    return FrameManifest(
        rgb_path=str(scene_dir / "rgb" / f"{int(im_id):06d}.png"),
        depth_path=str(scene_dir / "depth" / f"{int(im_id):06d}.png"),
        K=cam["cam_K"],
        depth_scale=float(cam.get("depth_scale", 1.0)) / 1000.0,
        scene_id=int(scene_id),
        im_id=int(im_id),
    )


def load_bop_scene(
    bop_root: str | os.PathLike,
    split: str,
    scene_id: int,
    im_id: int,
) -> Scene:
    """Load one BOP RGB-D frame as a ``Scene`` with depth in metres."""

    return load_scene_from_manifest(
        bop_frame_manifest(bop_root, split, scene_id, im_id)
    )


def _shard_path(shard_dir: str | os.PathLike, target: BOPTargetImage) -> Path:
    return (
        Path(shard_dir)
        / f"scene_{target.scene_id:06d}_im_{target.im_id:06d}.json"
    )


def _filter_target_objects(
    records: Sequence[dict],
    target: BOPTargetImage,
    enabled: bool,
) -> list[dict]:
    if not enabled:
        return [dict(rec) for rec in records]
    keep = set(target.obj_ids)
    return [dict(rec) for rec in records if int(rec["category_id"]) in keep]


def generate_bop_muse_detections(
    *,
    bop_root: str | os.PathLike,
    split: str,
    targets: Sequence[BOPTargetImage],
    segmentor,
    n_masks: int | None = 2,
    target_object_only: bool = False,
    shard_dir: str | os.PathLike | None = None,
    resume: bool = False,
    scene_loader: SceneLoader = load_bop_scene,
    record_time: bool = True,
    progress: ProgressCallback | None = None,
) -> list[dict]:
    """Run MUSE over target images and return BOP detections records.

    Per-image shards always store the full registered-class output. The final
    combined list may optionally filter to target objects, which keeps resume
    usable across full-output and target-only smoke runs.
    """

    all_records: list[dict] = []
    target_list = list(targets)
    if shard_dir is not None:
        Path(shard_dir).mkdir(parents=True, exist_ok=True)

    for idx, target in enumerate(target_list, start=1):
        shard = _shard_path(shard_dir, target) if shard_dir is not None else None
        resumed = bool(resume and shard is not None and shard.exists())
        if resumed:
            image_records = _read_json(shard)
            elapsed = float(image_records[0].get("time", 0.0)) if image_records else 0.0
        else:
            scene = scene_loader(bop_root, split, target.scene_id, target.im_id)
            start = time.perf_counter()
            image_records = muse_records(scene, segmentor, n_masks=n_masks)
            elapsed = time.perf_counter() - start
            if record_time:
                for rec in image_records:
                    rec["time"] = float(elapsed)
            if shard is not None:
                write_muse_detections(image_records, str(shard))

        emitted = _filter_target_objects(image_records, target, target_object_only)
        all_records.extend(emitted)
        if progress is not None:
            progress(idx, len(target_list), target, len(emitted), elapsed, resumed)
    return all_records


def _print_progress(
    idx: int,
    total: int,
    target: BOPTargetImage,
    n_records: int,
    elapsed: float,
    resumed: bool,
) -> None:
    state = "resume" if resumed else f"{elapsed:.2f}s"
    print(
        f"[{idx}/{total}] scene={target.scene_id:06d} im={target.im_id:06d} "
        f"targets={','.join(str(x) for x in target.obj_ids)} "
        f"records={n_records} {state}",
        flush=True,
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run MUSE over a BOP target split and write detections JSON "
                    "(source='muse-repro')."
    )
    ap.add_argument("--bop-root", required=True,
                    help="BOP dataset root, e.g. /workspace/bop_data/ycbv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--targets", default="",
                    help="BOP targets JSON; default: <bop-root>/<split>_targets_bop19.json")
    ap.add_argument("--models-info", default="",
                    help="default: <bop-root>/models/models_info.json")
    ap.add_argument("--models-dir", default="",
                    help="default: <bop-root>/models")
    ap.add_argument("--template-root", default="",
                    help="root containing one template directory per object")
    ap.add_argument("--template-pattern", default="obj_{obj_id:06d}",
                    help="format string under --template-root; receives obj_id")
    ap.add_argument("--classes", default="",
                    help="optional explicit 'obj_id=template_dir' entries. "
                         "Without --template-root or --objs, these ids also "
                         "define the object subset.")
    ap.add_argument("--objs", default="",
                    help="optional comma-separated object subset, e.g. 9,14")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-dir", default="",
                    help="optional per-image JSON directory for resume")
    ap.add_argument("--resume", action="store_true",
                    help="reuse existing per-image shards")
    ap.add_argument("--limit-images", type=int, default=0,
                    help="smoke subset; 0 means the whole target split")
    ap.add_argument("--target-object-only", action="store_true",
                    help="emit only objects listed in the BOP targets for each image. "
                         "Default keeps wrong-category detections as false positives.")
    ap.add_argument("--topk", type=int, default=2,
                    help="masks kept per class per image")
    ap.add_argument("--no-time", action="store_true",
                    help="do not stamp BOP per-detection runtime")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gdino", default=DEFAULT_GDINO_MODEL)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--box-threshold", type=float, default=0.15)
    ap.add_argument("--text-threshold", type=float, default=0.15)
    ap.add_argument("--sam2-size", default="large")
    ap.add_argument("--sam-ckpt-dir", default=None)
    ap.add_argument("--dinov2-model", default="dinov2_vitg14_reg")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--beta", type=float, default=DEFAULT_BETA)
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU)
    ap.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    ap.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    ap.add_argument("--allow-single-class", action="store_true")
    return ap.parse_args(argv)


def _main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    from popoe.segmentor_detections import load_bop_detections

    bop_root = Path(args.bop_root)
    targets_path = Path(args.targets) if args.targets else default_targets_path(bop_root, args.split)
    models_info = Path(args.models_info) if args.models_info else bop_root / "models" / "models_info.json"
    models_dir = Path(args.models_dir) if args.models_dir else bop_root / "models"
    obj_filter = _parse_obj_ids(args.objs)
    overrides = _parse_template_overrides(args.classes)
    if not obj_filter and overrides and not args.template_root:
        obj_filter = set(overrides)

    target_images = load_bop_target_images(
        targets_path,
        obj_ids=obj_filter or None,
        limit_images=args.limit_images,
    )
    if not target_images:
        raise SystemExit(f"no BOP target images after filtering: {targets_path}")

    target_obj_ids = sorted({oid for t in target_images for oid in t.obj_ids})
    classes = build_muse_classes(
        target_obj_ids,
        models_info=models_info,
        models_dir=models_dir,
        template_root=args.template_root or None,
        template_pattern=args.template_pattern,
        template_overrides=overrides,
    )
    seg = build_muse_segmentor(
        classes,
        device=args.device,
        gdino_model=args.gdino,
        prompt=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        sam2_size=args.sam2_size,
        sam_ckpt_dir=args.sam_ckpt_dir,
        dinov2_model=args.dinov2_model,
        min_pixels=args.min_pixels,
        alpha=args.alpha,
        beta=args.beta,
        tau=args.tau,
        gamma=args.gamma,
        n_masks=args.topk,
        allow_single_class=args.allow_single_class,
    )

    print(
        f"MUSE {MUSE_SOURCE}: {len(target_images)} images, "
        f"{len(classes)} classes -> {args.out}",
        flush=True,
    )
    records = generate_bop_muse_detections(
        bop_root=bop_root,
        split=args.split,
        targets=target_images,
        segmentor=seg,
        n_masks=args.topk,
        target_object_only=args.target_object_only,
        shard_dir=args.shard_dir or None,
        resume=args.resume,
        record_time=not args.no_time,
        progress=_print_progress,
    )
    write_muse_detections(records, args.out)
    load_bop_detections(args.out, source=MUSE_SOURCE)
    print(f"wrote {len(records)} detections -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "BOPTargetImage",
    "bop_frame_manifest",
    "build_muse_classes",
    "default_targets_path",
    "generate_bop_muse_detections",
    "load_bop_scene",
    "load_bop_target_images",
    "load_models_info",
]
