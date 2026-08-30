"""NIDS-Net adapter layer.

NIDS-Net itself is a heavy external producer (GroundingDINO + SAM + DINOv2
FFA/adapters, often Detectron2). popoe does not import it. The boundary here is
the artifact NIDS writes: 2D detections with mask, label, score, bbox. This
module stamps/normalises those records into popoe's detection schema and wraps
the generic file segmentor with a stable `source='nids'` provenance tag.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Mapping, Optional, Sequence

from popoe.segmentor_detections import load_detections


def _records_from_payload(payload):
    """Accept a COCO-result list or a dict carrying records under a common key."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("detections", "annotations", "instances", "results"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise TypeError("NIDS-Net output must be a JSON list or a dict with detections")


def _parse_id_map(spec: Optional[str]) -> dict:
    """Parse '1:9,2:10' into {1: 9, 2: 10}."""
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
    """Keep mask file references valid when adapted JSON is written elsewhere."""
    if "mask_path" in rec and isinstance(rec["mask_path"], str):
        rec["mask_path"] = _resolve_path(rec["mask_path"], base_dir)
    mask = rec.get("mask")
    if isinstance(mask, str):
        rec["mask"] = _resolve_path(mask, base_dir)
    elif isinstance(mask, dict) and "path" in mask and isinstance(mask["path"], str):
        rec["mask"] = dict(mask)
        rec["mask"]["path"] = _resolve_path(mask["path"], base_dir)
    return rec


def adapt_nidsnet_records(records,
                          *,
                          scene_id: Optional[int] = None,
                          image_id: Optional[int] = None,
                          category_id_map: Optional[Mapping] = None,
                          source: str = "nids",
                          base_dir: Optional[str] = None) -> list[dict]:
    """Normalise NIDS-Net/Detectron2-style detections for popoe.

    NIDS BOP predictions may already be popoe/BOP-compatible. Detectron2 COCO
    result files often have only `image_id/category_id/bbox/score/segmentation`;
    for real captures this function stamps the missing `scene_id` and can force
    one `image_id` from the frame manifest.

    The output is intentionally still a JSON-serialisable list of 2D detections.
    Depth never appears here; it belongs to `FrameManifest` / `Scene`.
    """
    out = []
    for i, raw in enumerate(_records_from_payload(records)):
        rec = dict(raw)
        if not _has_mask_record(rec):
            raise ValueError(
                f"NIDS record {i} has no mask. In the official NIDS-Net code, "
                "enable mask export before writing predictions.")

        if scene_id is not None:
            rec["scene_id"] = int(scene_id)
        elif "scene_id" not in rec:
            rec["scene_id"] = -1

        if image_id is not None:
            rec["image_id"] = int(image_id)
        elif "image_id" not in rec:
            rec["image_id"] = rec.get("im_id", -1)

        if "category_id" not in rec:
            raise KeyError(f"NIDS record {i} has no category_id")
        rec["category_id"] = _map_category(rec["category_id"], category_id_map)
        rec["score"] = float(rec.get("score", 1.0))
        if source:
            rec["source"] = source
        out.append(_resolve_mask_paths(rec, base_dir))
    return out


def adapt_nidsnet_json(input_json: str,
                       output_json: str,
                       *,
                       scene_id: Optional[int] = None,
                       image_id: Optional[int] = None,
                       category_id_map: Optional[Mapping] = None,
                       source: str = "nids") -> list[dict]:
    """Read NIDS-Net output, write popoe-compatible detections, return records."""
    with open(input_json) as f:
        payload = json.load(f)
    base_dir = os.path.dirname(os.path.abspath(input_json))
    records = adapt_nidsnet_records(
        payload,
        scene_id=scene_id,
        image_id=image_id,
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


def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Adapt NIDS-Net predictions into popoe detections JSON.")
    ap.add_argument("--input", required=True, help="raw NIDS-Net predictions JSON")
    ap.add_argument("--output", required=True, help="popoe detections JSON")
    ap.add_argument("--scene-id", type=int, default=None,
                    help="stamp scene_id when the input lacks one")
    ap.add_argument("--image-id", type=int, default=None,
                    help="stamp image_id for a single-frame prediction file")
    ap.add_argument("--category-map", default=None,
                    help="optional id map, e.g. '1:9,2:10'")
    ap.add_argument("--source", default="nids",
                    help="Detection.source provenance tag")
    args = ap.parse_args(argv)
    records = adapt_nidsnet_json(
        args.input,
        args.output,
        scene_id=args.scene_id,
        image_id=args.image_id,
        category_id_map=_parse_id_map(args.category_map),
        source=args.source,
    )
    # Load once through the production loader to catch schema errors now.
    load_detections(args.output, source=args.source)
    print(f"wrote {len(records)} detections -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "adapt_nidsnet_json",
    "adapt_nidsnet_records",
]
