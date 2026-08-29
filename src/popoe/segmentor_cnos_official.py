"""Official CNOS producer adapter.

The official `nv-nguyen/cnos` repository is a heavy external producer
(SAM/FastSAM + DINOv2 + Hydra). popoe does not import it. The boundary here is
the artifact it writes: BOP-style 2D detections JSON with masks, labels, boxes
and scores. This module provides a stable `source='cnos'` file segmentor and
small command builders for running the official repo in its own environment.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping, Optional, Sequence

from popoe.external_cmd import ExternalCommand, run_external_command
from popoe.segmentor_detections import BOPDetectionsSegmentor, load_detections


CNOS_SOURCE = "cnos"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_cnos_root(cnos_root: Optional[str] = None) -> Path:
    """Return the official CNOS checkout root.

    Priority:
    1. explicit argument;
    2. `POPOE_CNOS_PATH`;
    3. pinned submodule at `external/cnos`.
    """

    root = cnos_root or os.environ.get("POPOE_CNOS_PATH")
    if root is None:
        root = str(_repo_root() / "external" / "cnos")
    return Path(root).expanduser()


def _cnos_python(python: Optional[str] = None) -> str:
    return python or os.environ.get("POPOE_CNOS_PYTHON", "python")


def _model_override(model: Optional[str]) -> Optional[str]:
    if model is None:
        return None
    m = model.lower()
    if m in ("sam", "cnos"):
        return None
    if m in ("fastsam", "fast_sam", "cnos_fast"):
        return "model=cnos_fast"
    if "=" in model:
        return model
    return f"model={model}"


def build_cnos_bop_command(dataset: str,
                           *,
                           cnos_root: Optional[str] = None,
                           python: Optional[str] = None,
                           model: Optional[str] = "fastsam",
                           rendering_type: Optional[str] = "pbr",
                           level_templates: Optional[int] = None,
                           gpu: Optional[str] = None,
                           overrides: Sequence[str] = ()) -> ExternalCommand:
    """Build the official CNOS BOP inference command."""

    argv = [_cnos_python(python), "run_inference.py", f"dataset_name={dataset}"]
    model_arg = _model_override(model)
    if model_arg:
        argv.append(model_arg)
    if rendering_type:
        argv.append(f"model.onboarding_config.rendering_type={rendering_type}")
    if level_templates is not None:
        argv.append(f"model.onboarding_config.level_templates={int(level_templates)}")
    argv.extend(overrides)
    env = {}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return ExternalCommand(tuple(argv), cwd=str(resolve_cnos_root(cnos_root)), env=env)


def build_cnos_custom_render_command(cad_path: str,
                                     output_dir: str,
                                     *,
                                     cnos_root: Optional[str] = None,
                                     rgb_path: Optional[str] = None) -> ExternalCommand:
    """Build the official custom-template rendering command.

    The upstream script reads CAD_PATH and OUTPUT_DIR from the environment.
    RGB_PATH is not needed for rendering, but passing it keeps the env block
    symmetric with the custom inference command.
    """

    env = {"CAD_PATH": cad_path, "OUTPUT_DIR": output_dir}
    if rgb_path is not None:
        env["RGB_PATH"] = rgb_path
    return ExternalCommand(
        ("bash", "./src/scripts/render_custom.sh"),
        cwd=str(resolve_cnos_root(cnos_root)),
        env=env,
    )


def build_cnos_custom_infer_command(rgb_path: str,
                                    output_dir: str,
                                    *,
                                    cnos_root: Optional[str] = None,
                                    python: Optional[str] = None,
                                    num_max_dets: int = 1,
                                    conf_threshold: float = 0.5,
                                    stability_score_thresh: float = 0.5) -> ExternalCommand:
    """Build the official custom RGB/CAD inference command."""

    argv = [
        _cnos_python(python),
        "-m",
        "src.scripts.inference_custom",
        "--template_dir",
        output_dir,
        "--rgb_path",
        rgb_path,
        "--num_max_dets",
        str(int(num_max_dets)),
        "--confg_threshold",  # upstream spelling
        str(float(conf_threshold)),
        "--stability_score_thresh",
        str(float(stability_score_thresh)),
    ]
    return ExternalCommand(tuple(argv), cwd=str(resolve_cnos_root(cnos_root)))


def _records_from_payload(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("detections", "annotations", "instances", "results"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise TypeError("CNOS output must be a JSON list or a dict with detections")


def _parse_id_map(spec: Optional[str]) -> dict:
    if not spec:
        return {}
    out = {}
    for part in spec.split(","):
        if not part.strip():
            continue
        left, right = part.split(":", 1)
        out[int(left.strip())] = int(right.strip())
    return out


def _map_category(category_id, category_id_map: Optional[Mapping]) -> int:
    cid = int(float(category_id))
    if not category_id_map:
        return cid
    if cid in category_id_map:
        return int(category_id_map[cid])
    key = str(cid)
    if key in category_id_map:
        return int(category_id_map[key])
    return cid


def _has_mask_record(rec: dict) -> bool:
    return any(k in rec for k in ("segmentation", "mask", "mask_path"))


def _resolve_path(path: str, base_dir: Optional[str]) -> str:
    if base_dir is None or os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def _resolve_mask_paths(rec: dict, base_dir: Optional[str]) -> dict:
    if "mask_path" in rec and isinstance(rec["mask_path"], str):
        rec["mask_path"] = _resolve_path(rec["mask_path"], base_dir)
    mask = rec.get("mask")
    if isinstance(mask, str):
        rec["mask"] = _resolve_path(mask, base_dir)
    elif isinstance(mask, dict) and "path" in mask and isinstance(mask["path"], str):
        rec["mask"] = dict(mask)
        rec["mask"]["path"] = _resolve_path(mask["path"], base_dir)
    return rec


def adapt_cnos_records(records,
                       *,
                       scene_id: Optional[int] = None,
                       image_id: Optional[int] = None,
                       category_id: Optional[int] = None,
                       category_id_map: Optional[Mapping] = None,
                       source: str = CNOS_SOURCE,
                       base_dir: Optional[str] = None) -> list[dict]:
    """Stamp/remap official CNOS detections for popoe consumption.

    Official BOP outputs already carry authoritative ids. The custom CAD/RGB
    script writes placeholder ids, so single-frame custom output must be
    stamped with the target frame/object ids before using `CNOSDetectionsSegmentor`.
    """

    out = []
    for i, raw in enumerate(_records_from_payload(records)):
        rec = dict(raw)
        if not _has_mask_record(rec):
            raise ValueError(f"CNOS record {i} has no mask")

        if scene_id is not None:
            rec["scene_id"] = int(scene_id)
        elif "scene_id" not in rec:
            rec["scene_id"] = -1

        if image_id is not None:
            rec["image_id"] = int(image_id)
        elif "image_id" not in rec:
            rec["image_id"] = rec.get("im_id", -1)

        if category_id is not None:
            rec["category_id"] = int(category_id)
        elif "category_id" in rec:
            rec["category_id"] = _map_category(rec["category_id"], category_id_map)
        else:
            raise KeyError(f"CNOS record {i} has no category_id")

        rec["score"] = float(rec.get("score", 1.0))
        if source:
            rec["source"] = source
        out.append(_resolve_mask_paths(rec, base_dir))
    return out


def adapt_cnos_json(input_json: str,
                    output_json: str,
                    *,
                    scene_id: Optional[int] = None,
                    image_id: Optional[int] = None,
                    category_id: Optional[int] = None,
                    category_id_map: Optional[Mapping] = None,
                    source: str = CNOS_SOURCE) -> list[dict]:
    """Read official CNOS output, write popoe-compatible detections JSON."""

    with open(input_json) as f:
        payload = json.load(f)
    base_dir = os.path.dirname(os.path.abspath(input_json))
    records = adapt_cnos_records(
        payload,
        scene_id=scene_id,
        image_id=image_id,
        category_id=category_id,
        category_id_map=category_id_map,
        source=source,
        base_dir=base_dir,
    )
    out_dir = os.path.dirname(os.path.abspath(output_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(records, f)
    return records


class CNOSDetectionsSegmentor(BOPDetectionsSegmentor):
    """File-backed Segmentor for official CNOS/CNOS-FastSAM predictions."""

    source = CNOS_SOURCE


def _add_root_args(p):
    p.add_argument("--cnos-root", default=None,
                   help="official CNOS checkout; default POPOE_CNOS_PATH or external/cnos")
    p.add_argument("--python", default=None,
                   help="python executable in the CNOS environment")


def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build official CNOS external commands or inspect detections.")
    sub = ap.add_subparsers(dest="command", required=True)

    infer = sub.add_parser("infer-command", help="print official CNOS BOP command")
    _add_root_args(infer)
    infer.add_argument("--dataset", required=True)
    infer.add_argument("--model", default="fastsam",
                       help="'fastsam', 'sam', or a Hydra model override")
    infer.add_argument("--rendering-type", default="pbr",
                       help="Hydra rendering type, e.g. pbr or pyrender")
    infer.add_argument("--level-templates", type=int, default=None)
    infer.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value")
    infer.add_argument("--override", action="append", default=[],
                       help="additional Hydra override, repeatable")

    render = sub.add_parser("custom-render-command",
                            help="print official custom CAD template render command")
    render.add_argument("--cnos-root", default=None)
    render.add_argument("--cad-path", required=True)
    render.add_argument("--rgb-path", default=None)
    render.add_argument("--output-dir", required=True)

    custom = sub.add_parser("custom-infer-command",
                            help="print official custom RGB inference command")
    _add_root_args(custom)
    custom.add_argument("--rgb-path", required=True)
    custom.add_argument("--output-dir", required=True)
    custom.add_argument("--num-max-dets", type=int, default=1)
    custom.add_argument("--conf-threshold", type=float, default=0.5)
    custom.add_argument("--stability-score-thresh", type=float, default=0.5)

    adapt = sub.add_parser("adapt-custom",
                           help="stamp/remap official CNOS custom detections")
    adapt.add_argument("--input", required=True, help="official CNOS detection.json")
    adapt.add_argument("--output", required=True, help="popoe detections JSON")
    adapt.add_argument("--scene-id", type=int, default=None,
                       help="force scene_id for a single-frame custom output")
    adapt.add_argument("--image-id", type=int, default=None,
                       help="force image_id for a single-frame custom output")
    adapt.add_argument("--category-id", type=int, default=None,
                       help="force one object id for a single-CAD custom output")
    adapt.add_argument("--category-map", default=None,
                       help="optional id map using JSON category_id keys as "
                            "written by the producer; official custom uses "
                            "object_ids+1 so single-CAD is usually '1:9'")
    adapt.add_argument("--source", default=CNOS_SOURCE,
                       help="Detection.source provenance tag")

    check = sub.add_parser("check", help="load a CNOS detections JSON")
    check.add_argument("--input", required=True)

    args = ap.parse_args(argv)
    if args.command == "infer-command":
        cmd = build_cnos_bop_command(
            args.dataset,
            cnos_root=args.cnos_root,
            python=args.python,
            model=args.model,
            rendering_type=args.rendering_type,
            level_templates=args.level_templates,
            gpu=args.gpu,
            overrides=args.override,
        )
        print(cmd.shell_line())
        return 0
    if args.command == "custom-render-command":
        cmd = build_cnos_custom_render_command(
            args.cad_path,
            args.output_dir,
            cnos_root=args.cnos_root,
            rgb_path=args.rgb_path,
        )
        print(cmd.shell_line())
        return 0
    if args.command == "custom-infer-command":
        cmd = build_cnos_custom_infer_command(
            args.rgb_path,
            args.output_dir,
            cnos_root=args.cnos_root,
            python=args.python,
            num_max_dets=args.num_max_dets,
            conf_threshold=args.conf_threshold,
            stability_score_thresh=args.stability_score_thresh,
        )
        print(cmd.shell_line())
        return 0
    if args.command == "adapt-custom":
        if (args.scene_id is None and args.image_id is None
                and args.category_id is None and not args.category_map):
            raise SystemExit(
                "adapt-custom requires at least one of "
                "--scene-id / --image-id / --category-id / --category-map "
                "(official custom leaves scene_id=0, image_id=0, category_id=1)")
        records = adapt_cnos_json(
            args.input,
            args.output,
            scene_id=args.scene_id,
            image_id=args.image_id,
            category_id=args.category_id,
            category_id_map=_parse_id_map(args.category_map),
            source=args.source,
        )
        load_detections(args.output, source=args.source)
        print(f"wrote {len(records)} CNOS detections -> {args.output}")
        return 0

    records = load_detections(args.input, source=CNOS_SOURCE)
    print(f"loaded {len(records)} CNOS detections from {args.input}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "CNOSDetectionsSegmentor",
    "ExternalCommand",
    "adapt_cnos_json",
    "adapt_cnos_records",
    "build_cnos_bop_command",
    "build_cnos_custom_infer_command",
    "build_cnos_custom_render_command",
    "resolve_cnos_root",
    "run_external_command",
]
