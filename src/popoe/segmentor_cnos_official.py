"""Official CNOS producer adapter.

The official `nv-nguyen/cnos` repository is a heavy external producer
(SAM/FastSAM + DINOv2 + Hydra). popoe does not import it. The boundary here is
the artifact it writes: BOP-style 2D detections JSON with masks, labels, boxes
and scores. This module provides a stable `source='cnos'` file segmentor and
small command builders for running the official repo in its own environment.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

from popoe.segmentor_detections import BOPDetectionsSegmentor, load_detections


CNOS_SOURCE = "cnos"


@dataclass(frozen=True)
class ExternalCommand:
    """A subprocess command plus the cwd/env needed by the official repo."""

    argv: tuple[str, ...]
    cwd: str
    env: Mapping[str, str] = field(default_factory=dict)

    def as_subprocess_kwargs(self) -> dict:
        env = os.environ.copy()
        env.update(dict(self.env))
        return {"args": list(self.argv), "cwd": self.cwd, "env": env}

    def shell_line(self) -> str:
        env_prefix = " ".join(
            f"{k}={shlex.quote(v)}" for k, v in sorted(self.env.items())
        )
        if env_prefix:
            env_prefix += " "
        argv = " ".join(shlex.quote(a) for a in self.argv)
        return f"cd {shlex.quote(self.cwd)} && {env_prefix}{argv}"


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


def run_external_command(command: ExternalCommand, **kwargs) -> subprocess.CompletedProcess:
    popen_kwargs = command.as_subprocess_kwargs()
    popen_kwargs.update(kwargs)
    return subprocess.run(**popen_kwargs)


class CNOSDetectionsSegmentor(BOPDetectionsSegmentor):
    """File-backed Segmentor for official CNOS/CNOS-FastSAM predictions."""

    source = CNOS_SOURCE

    def __init__(self, detections_json: str, topk: int = 2,
                 merge_labels: Optional[dict] = None,
                 iou_dedupe: float = 0.9,
                 min_pixels: int = 100,
                 source: str = CNOS_SOURCE):
        super().__init__(
            detections_json,
            topk=topk,
            merge_labels=merge_labels,
            iou_dedupe=iou_dedupe,
            min_pixels=min_pixels,
            source=source,
        )


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

    records = load_detections(args.input, source=CNOS_SOURCE)
    print(f"loaded {len(records)} CNOS detections from {args.input}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "CNOSDetectionsSegmentor",
    "ExternalCommand",
    "build_cnos_bop_command",
    "build_cnos_custom_infer_command",
    "build_cnos_custom_render_command",
    "resolve_cnos_root",
    "run_external_command",
]
