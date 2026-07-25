# NIDS-Net Deployment Notes

NIDS-Net is a detector/segmentor producer for popoe. It should run in its own
environment or service and write a detections JSON; popoe then consumes that
file through `popoe.segmentor_nids.NIDSNetDetectionsSegmentor` or the generic
multi-source union.

The official source is pinned as a submodule at `external/NIDS-Net`:

```bash
git submodule update --init --recursive external/NIDS-Net
git -C external/NIDS-Net rev-parse HEAD
# c7685a442157a1f28f2d7771e10dd9c7afdd7154
```

Use the submodule for source provenance and local deployment notes; keep the
runtime environment separate from popoe.

## Boundary

Keep the dependency boundary strict:

```text
NIDS-Net env/service
  RGB/templates -> 2D detections JSON

popoe env/service
  RGB-D frame manifest + detections JSON -> 6D pose
```

The detections JSON carries only 2D information: `scene_id`, `image_id`,
`category_id`, `score`, `bbox`, and `mask`/`segmentation`. Depth stays in the
frame manifest and is loaded into `Scene.depth` in metres.

## Why A Separate Environment

The official NIDS-Net repository uses GroundingDINO + SAM proposals, DINOv2
foreground feature averaging/adapters, and often Detectron2 plus version-pinned
support packages. Those dependencies are much more volatile than the pose
backend dependencies (`open3d`, GeDi/dGeDi, nvdiffrast). Put NIDS-Net in a
separate `uv` or conda environment and exchange JSON files or HTTP payloads.

On a 4090 host this is fine: the environments do not conflict at runtime, but
the processes still share GPU memory. For a single-GPU workstation, prefer
serial execution: run NIDS, release the model/process, then run popoe pose.

## BOP Mode

Prefer published prediction files when available. For popoe evaluation they are
just another named source:

```python
from popoe.segmentor_detections import BOPDetectionsSegmentor

seg = BOPDetectionsSegmentor(sources={
    "cnos": "data/detections/cnos/cnos-fastsam_ycbv-test.json",
    "nids": "data/detections/nids/nids_wa_sappe_ycbv.json",
}, topk=2)
```

If you need to regenerate NIDS predictions, use the pinned `external/NIDS-Net`
checkout in its own environment. Its README documents the BOP path as
`python run_inference.py dataset_name=<dataset>` after downloading template
embeddings and adapter weights. Once a prediction JSON exists, verify it
through popoe's loader:

```bash
python - <<'PY'
from popoe.segmentor_detections import load_detections, decode_detection_mask
p = "data/detections/nids/nids_wa_sappe_lmo.json"
d = load_detections(p, source="nids")[0]
m = decode_detection_mask(d["segmentation"])
print(d["scene_id"], d["image_id"], d["category_id"], d["score"], m.shape, m.dtype)
PY
```

## Real Scene Mode

For a real frame, save a frame manifest:

```json
{
  "scene_id": 0,
  "image_id": 42,
  "rgb_path": "frames/000042_rgb.png",
  "depth_path": "frames/000042_depth.png",
  "depth_scale": 0.001,
  "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "detections_path": "detections/000042_nids.json"
}
```

If the raw NIDS output is already BOP-like and includes masks, popoe can read it
directly. If it is Detectron2/COCO-style or missing `scene_id`, adapt it:

```bash
popoe-nids-adapt \
  --input outputs/nids_raw_000042.json \
  --output detections/000042_nids.json \
  --scene-id 0 \
  --image-id 42 \
  --category-map 1:9,2:10
```

The module form is equivalent when running from a checkout:

```bash
python -m popoe.segmentor_nids --input ... --output ...
```

Then consume it:

```python
from popoe.datasets.frames import load_frame_manifest, load_scene_from_manifest
from popoe.segmentor_nids import NIDSNetDetectionsSegmentor

frame = load_frame_manifest("captures/frame_000042.json")
scene = load_scene_from_manifest(frame)
seg = NIDSNetDetectionsSegmentor(frame.detections_path, topk=2)
dets = seg.segment(scene, obj)
```

## Service Shape

For a long-running deployment, make NIDS-Net a detector service whose response
body is the same list written to `detections_path`. A minimal API is:

```text
POST /detect
request:  { "rgb_path": "...", "scene_id": 0, "image_id": 42,
            "object_set": "ycbv", "templates_path": "..." }
response: [ { "scene_id": 0, "image_id": 42, "category_id": 9,
              "score": 0.91, "bbox": [x, y, w, h],
              "mask": {"format": "rle", "size": [H, W], "counts": "..."} } ]
```

popoe's pose service should not import NIDS-Net. It should accept a frame
manifest plus detections, then run the pose pipeline.

## Operational Checks

- Confirm masks are present. The official NIDS-Net README notes that mask export
  may need to be enabled in the prediction path.
- Confirm object IDs match the CAD set consumed by popoe.
- Confirm `image_id`/`scene_id` match the frame manifest.
- Confirm `depth_scale` converts raw depth to metres.
- Keep NIDS source names explicit: use `source="nids"` or
  `sources={"nids": path}` so provenance survives into scoring and logs.
