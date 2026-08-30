"""SAM-6D adapter layer.

SAM-6D is treated as a heavy external producer. popoe does not import the
official ISM/PEM packages; it consumes their artifacts:

* ISM -> BOP-style 2D detections JSON, wrapped as `source='sam6d'`.
* PEM -> pose predictions, either BOP CSV or custom JSON with R/t fields,
  wrapped as `PoseHypothesis` values.

The command builders below return the exact subprocess boundary to run in the
separate SAM-6D environment. They do not execute unless the caller asks.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from popoe.external_cmd import ExternalCommand
from popoe.interfaces import Detection, ObjectModel, PoseHypothesis, Scene



SAM6D_SOURCE = "sam6d"
SAM6D_PEM_SOURCE = "sam6d-pem"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_sam6d_root(sam6d_root: Optional[str] = None) -> Path:
    """Return the official SAM-6D checkout root.

    Priority:
    1. explicit argument;
    2. `POPOE_SAM6D_PATH`;
    3. pinned submodule at `external/SAM-6D`.
    """

    root = sam6d_root or os.environ.get("POPOE_SAM6D_PATH")
    if root is None:
        root = str(_repo_root() / "external" / "SAM-6D")
    return Path(root).expanduser()


def _sam6d_python(python: Optional[str] = None) -> str:
    return python or os.environ.get("POPOE_SAM6D_PYTHON", "python")


def _ism_dir(sam6d_root: Optional[str] = None) -> Path:
    return resolve_sam6d_root(sam6d_root) / "SAM-6D" / "Instance_Segmentation_Model"


def _pem_dir(sam6d_root: Optional[str] = None) -> Path:
    return resolve_sam6d_root(sam6d_root) / "SAM-6D" / "Pose_Estimation_Model"


def _ism_model_override(model: Optional[str]) -> Optional[str]:
    if not model or model.lower() == "sam":
        return None
    if "=" in model:
        return model
    if model.lower() == "fastsam":
        return "model=ISM_fastsam"
    return f"model={model}"


def build_ism_bop_command(dataset: str,
                          *,
                          sam6d_root: Optional[str] = None,
                          python: Optional[str] = None,
                          model: Optional[str] = None,
                          gpu: Optional[str] = None,
                          overrides: Sequence[str] = ()) -> ExternalCommand:
    """Build the official SAM-6D ISM BOP inference command.

    The command must run from `SAM-6D/Instance_Segmentation_Model` because the
    upstream Hydra config and log paths are relative to that directory.
    """

    argv = [_sam6d_python(python), "run_inference.py", f"dataset_name={dataset}"]
    model_override = _ism_model_override(model)
    if model_override:
        argv.append(model_override)
    argv.extend(overrides)
    env = {}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return ExternalCommand(tuple(argv), cwd=str(_ism_dir(sam6d_root)), env=env)


def build_pem_bop_command(dataset: str,
                          *,
                          sam6d_root: Optional[str] = None,
                          python: Optional[str] = None,
                          gpus: str = "0",
                          model: str = "pose_estimation_model",
                          config: str = "config/base.yaml",
                          checkpoint_path: Optional[str] = None,
                          view: Optional[int] = 42,
                          exp_id: Optional[int] = None,
                          extra_args: Sequence[str] = ()) -> ExternalCommand:
    """Build the official SAM-6D PEM BOP inference command."""

    argv = [
        _sam6d_python(python),
        "test_bop.py",
        "--gpus",
        str(gpus),
        "--model",
        model,
        "--config",
        config,
        "--dataset",
        dataset,
    ]
    if checkpoint_path:
        argv.extend(["--checkpoint_path", checkpoint_path])
    if view is not None:
        argv.extend(["--view", str(view)])
    if exp_id is not None:
        argv.extend(["--exp_id", str(exp_id)])
    argv.extend(extra_args)
    return ExternalCommand(tuple(argv), cwd=str(_pem_dir(sam6d_root)))


@dataclass(frozen=True)
class SAM6DPemPrediction:
    """One SAM-6D PEM pose row, keyed like BOP targets."""

    scene_id: int
    im_id: int
    obj_id: int
    hypothesis: PoseHypothesis


def _parse_scalar(x, *, name: str) -> float:
    s = str(x).strip()
    if not s:
        raise ValueError(f"{name} is empty")
    # Avoid np.fromstring for scalar fields: on non-numeric headers it emits a
    # DeprecationWarning before raising. BOP CSV headers should be skipped
    # cleanly by the caller's ValueError path.
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    try:
        return float(s)
    except ValueError as e:
        raise ValueError(f"{name} must be a float, got {x!r}") from e


def _parse_int(x, *, name: str) -> int:
    f = _parse_scalar(x, name=name)
    if not float(f).is_integer():
        raise ValueError(f"{name} must be integer-valued, got {x!r}")
    return int(f)


def _parse_vec(x, n: int, *, name: str) -> np.ndarray:
    text = str(x).replace("[", " ").replace("]", " ").replace(";", " ")
    vals = np.fromstring(text, sep=" ", dtype=np.float64)
    if vals.size != n:
        raise ValueError(f"{name} must contain {n} floats, got {x!r}")
    return vals


def _pose_prediction(scene_id, im_id, obj_id, score, R, t, time_s,
                     *, translation_scale: float,
                     source: str) -> SAM6DPemPrediction:
    score_f = _parse_scalar(score, name="score")
    R_arr = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t_arr = np.asarray(t, dtype=np.float64).reshape(3) * float(translation_scale)
    breakdown = {
        "source": source,
        "pem_score": score_f,
        "time": float(time_s),
    }
    return SAM6DPemPrediction(
        scene_id=_parse_int(scene_id, name="scene_id"),
        im_id=_parse_int(im_id, name="im_id"),
        obj_id=_parse_int(obj_id, name="obj_id"),
        hypothesis=PoseHypothesis(R=R_arr, t=t_arr, score=score_f,
                                  breakdown=breakdown),
    )


def load_sam6d_pem_csv(path: str,
                       *,
                       translation_scale: float = 0.001,
                       source: str = SAM6D_PEM_SOURCE) -> list[SAM6DPemPrediction]:
    """Load SAM-6D PEM BOP CSV output.

    BOP result translations are in millimetres, so the default
    `translation_scale=0.001` converts `PoseHypothesis.t` to metres.
    """

    out: list[SAM6DPemPrediction] = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for lineno, row in enumerate(reader, 1):
            if not row or all(not c.strip() for c in row):
                continue
            try:
                scene_id = _parse_int(row[0], name="scene_id")
            except ValueError:
                if lineno == 1:
                    continue
                raise
            if len(row) < 6:
                raise ValueError(f"{path}:{lineno}: expected at least 6 columns")
            time_s = _parse_scalar(row[6], name="time") if len(row) > 6 else 0.0
            out.append(_pose_prediction(
                scene_id,
                row[1],
                row[2],
                row[3],
                _parse_vec(row[4], 9, name="R").reshape(3, 3),
                _parse_vec(row[5], 3, name="t"),
                time_s,
                translation_scale=translation_scale,
                source=source,
            ))
    return out


def _records_from_json_payload(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("detections", "results", "predictions", "instances"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise TypeError("SAM-6D PEM JSON must be a list or dict carrying records")


def _json_id(rec: dict, keys: Sequence[str], override: Optional[int],
             *, name: str, record_index: int):
    raw = None
    for key in keys:
        if key in rec:
            raw = rec[key]
            break
    if override is None:
        return -1 if raw is None else raw
    if raw is None:
        return int(override)
    raw_id = _parse_int(raw, name=name)
    if raw_id in (-1, 0) or raw_id == int(override):
        return int(override)
    raise ValueError(
        f"SAM-6D PEM JSON record {record_index} has non-placeholder "
        f"{name}={raw_id}; refusing to stamp override {override}"
    )


def load_sam6d_pem_json(path: str,
                        *,
                        scene_id: Optional[int] = None,
                        image_id: Optional[int] = None,
                        obj_id: Optional[int] = None,
                        translation_scale: float = 0.001,
                        source: str = SAM6D_PEM_SOURCE) -> list[SAM6DPemPrediction]:
    """Load SAM-6D custom PEM JSON output with `R` and `t` fields.

    `scene_id`, `image_id`, and `obj_id` stamp missing/placeholder ids
    (`-1`/`0`) from single-frame custom demos. They are not filters; a
    conflicting non-placeholder record id raises instead of being collapsed.
    """

    with open(path) as f:
        payload = json.load(f)
    out: list[SAM6DPemPrediction] = []
    for i, rec in enumerate(_records_from_json_payload(payload)):
        if "R" not in rec or "t" not in rec:
            raise KeyError(f"SAM-6D PEM JSON record {i} needs R and t")
        sid = _json_id(rec, ("scene_id",), scene_id,
                       name="scene_id", record_index=i)
        iid = _json_id(rec, ("image_id", "im_id"), image_id,
                       name="image_id", record_index=i)
        oid = _json_id(rec, ("obj_id", "category_id"), obj_id,
                       name="obj_id", record_index=i)
        out.append(_pose_prediction(
            sid,
            iid,
            oid,
            rec.get("score", 1.0),
            rec["R"],
            rec["t"],
            rec.get("time", 0.0),
            translation_scale=translation_scale,
            source=source,
        ))
    return out


def load_sam6d_pem_results(path: str,
                           *,
                           scene_id: Optional[int] = None,
                           image_id: Optional[int] = None,
                           obj_id: Optional[int] = None,
                           translation_scale: float = 0.001,
                           source: str = SAM6D_PEM_SOURCE) -> list[SAM6DPemPrediction]:
    """Load PEM pose output from BOP CSV or SAM-6D custom JSON."""

    if Path(path).suffix.lower() == ".json":
        return load_sam6d_pem_json(
            path,
            scene_id=scene_id,
            image_id=image_id,
            obj_id=obj_id,
            translation_scale=translation_scale,
            source=source,
        )
    if scene_id is not None or image_id is not None or obj_id is not None:
        raise ValueError(
            "scene_id/image_id/obj_id overrides are only supported for SAM-6D "
            "PEM JSON outputs; CSV results must carry authoritative BOP ids"
        )
    return load_sam6d_pem_csv(
        path,
        translation_scale=translation_scale,
        source=source,
    )


class SAM6DPemResultsCoarseEstimator:
    """`CoarseEstimator` backed by already-written SAM-6D PEM results."""

    source = SAM6D_PEM_SOURCE

    def __init__(self, results_path: str,
                 *,
                 topk: int = 1,
                 scene_id: Optional[int] = None,
                 image_id: Optional[int] = None,
                 obj_id: Optional[int] = None,
                 translation_scale: float = 0.001,
                 source: str = SAM6D_PEM_SOURCE):
        self.results_path = results_path
        self.topk = int(topk)
        self.source = source
        self.predictions = load_sam6d_pem_results(
            results_path,
            scene_id=scene_id,
            image_id=image_id,
            obj_id=obj_id,
            translation_scale=translation_scale,
            source=source,
        )
        self._by_key: dict[tuple[int, int, int], list[PoseHypothesis]] = {}
        for pred in self.predictions:
            self._by_key.setdefault(
                (pred.scene_id, pred.im_id, pred.obj_id), []
            ).append(pred.hypothesis)
        for hyps in self._by_key.values():
            hyps.sort(key=lambda h: -h.score)

    def estimate(self, scene: Scene, obj: ObjectModel,
                 det: Optional[Detection] = None) -> list[PoseHypothesis]:
        del det
        key = (scene.scene_id, scene.im_id, obj.obj_id)
        return list(self._by_key.get(key, []))[: self.topk]


def _add_root_args(p):
    p.add_argument("--sam6d-root", default=None,
                   help="official SAM-6D checkout; default POPOE_SAM6D_PATH or external/SAM-6D")
    p.add_argument("--python", default=None,
                   help="python executable in the SAM-6D environment")


def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build SAM-6D external commands or inspect PEM results.")
    sub = ap.add_subparsers(dest="command", required=True)

    ism = sub.add_parser("ism-command", help="print the official ISM BOP command")
    _add_root_args(ism)
    ism.add_argument("--dataset", required=True)
    ism.add_argument("--model", default=None, help="'sam', 'fastsam', or a Hydra model override")
    ism.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value")
    ism.add_argument("--override", action="append", default=[],
                     help="additional Hydra override, repeatable")

    pem = sub.add_parser("pem-command", help="print the official PEM BOP command")
    _add_root_args(pem)
    pem.add_argument("--dataset", required=True)
    pem.add_argument("--gpus", default="0")
    pem.add_argument("--model", default="pose_estimation_model")
    pem.add_argument("--config", default="config/base.yaml")
    pem.add_argument("--checkpoint-path", default=None)
    pem.add_argument("--view", type=int, default=42)
    pem.add_argument("--exp-id", type=int, default=None)
    pem.add_argument("--extra-arg", action="append", default=[],
                     help="extra test_bop.py arg token, repeatable")

    chk = sub.add_parser("pem-check", help="load PEM CSV/JSON and print count")
    chk.add_argument("--input", required=True)
    chk.add_argument("--scene-id", type=int, default=None)
    chk.add_argument("--image-id", type=int, default=None)
    chk.add_argument("--obj-id", type=int, default=None)

    args = ap.parse_args(argv)
    if args.command == "ism-command":
        cmd = build_ism_bop_command(
            args.dataset,
            sam6d_root=args.sam6d_root,
            python=args.python,
            model=args.model,
            gpu=args.gpu,
            overrides=args.override,
        )
        print(cmd.shell_line())
        return 0
    if args.command == "pem-command":
        cmd = build_pem_bop_command(
            args.dataset,
            sam6d_root=args.sam6d_root,
            python=args.python,
            gpus=args.gpus,
            model=args.model,
            config=args.config,
            checkpoint_path=args.checkpoint_path,
            view=args.view,
            exp_id=args.exp_id,
            extra_args=args.extra_arg,
        )
        print(cmd.shell_line())
        return 0

    preds = load_sam6d_pem_results(
        args.input,
        scene_id=args.scene_id,
        image_id=args.image_id,
        obj_id=args.obj_id,
    )
    print(f"loaded {len(preds)} SAM-6D PEM predictions from {args.input}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "ExternalCommand",
    "SAM6D_SOURCE",
    "SAM6DPemPrediction",
    "SAM6DPemResultsCoarseEstimator",
    "build_ism_bop_command",
    "build_pem_bop_command",
    "load_sam6d_pem_csv",
    "load_sam6d_pem_json",
    "load_sam6d_pem_results",
    "resolve_sam6d_root",
]
