# NIDS-Net

NIDS-Net is a detection producer. popoe reads the detections JSON it writes through `BOPDetectionsSegmentor(..., source="nids")`; there is no `NIDSNetDetectionsSegmentor`. The producer boundary, the shared checks, and `BOPDetectionsSegmentor` usage are described in [README.md](README.md).

## Upstream

The official source is pinned as a submodule:

```bash
git submodule update --init --recursive external/NIDS-Net
git -C external/NIDS-Net rev-parse HEAD
```

The official repository combines Grounding DINO + SAM proposals, DINOv2 foreground feature averaging and adapters, and in most configurations Detectron2 with version-pinned support packages. Keep that runtime separate from popoe.

## BOP mode

Prefer the published prediction files. Download and hashes: [`data/detections/nids/PROVENANCE.md`](../../data/detections/nids/PROVENANCE.md).

```python
from popoe.segmentor_detections import BOPDetectionsSegmentor

seg = BOPDetectionsSegmentor(
    "data/detections/nids/nids_wa_sappe_ycbv.json", source="nids", topk=2
)
```

To regenerate them, use the pinned `external/NIDS-Net` checkout in its own environment. Its README documents the BOP path as `python run_inference.py dataset_name=<dataset>` after downloading template embeddings and adapter weights.

Verify a prediction JSON through popoe's loader:

```bash
python - <<'PY'
from popoe.segmentor_detections import load_detections, decode_detection_mask
p = "data/detections/nids/nids_wa_sappe_lmo.json"
d = load_detections(p, source="nids")[0]
m = decode_detection_mask(d["segmentation"])
print(d["scene_id"], d["image_id"], d["category_id"], d["score"], m.shape, m.dtype)
PY
```

Mask export may have to be enabled explicitly in the official prediction path. A boxes-only file loads successfully but contains no masks.

## Real scene mode

Save a frame manifest for the capture:

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

If the raw NIDS output is already in BOP form and includes masks, popoe reads it directly. If it is Detectron2/COCO-style, or lacks `scene_id`, normalise it first with `popoe-nids-adapt` (`popoe.segmentor_nids.adapt_nidsnet_json`):

```bash
popoe-nids-adapt \
  --input outputs/nids_raw_000042.json \
  --output detections/000042_nids.json \
  --scene-id 0 --image-id 42 --category-map 1:9,2:10
```

From a checkout, `python -m popoe.segmentor_nids --input ... --output ...` is equivalent.

Then consume it:

```python
from popoe.datasets.frames import load_frame_manifest, load_scene_from_manifest
from popoe.segmentor_detections import BOPDetectionsSegmentor

frame = load_frame_manifest("captures/frame_000042.json")
scene = load_scene_from_manifest(frame)
seg = BOPDetectionsSegmentor(frame.detections_path, source="nids", topk=2)
dets = seg.segment(scene, obj)
```

Confirm that `depth_scale` converts raw depth to metres.

## Service shape

For a long-running deployment, make NIDS-Net a detector service whose response body is the same list written to `detections_path`:

```text
POST /detect
request:  { "rgb_path": "...", "scene_id": 0, "image_id": 42,
            "object_set": "ycbv", "templates_path": "..." }
response: [ { "scene_id": 0, "image_id": 42, "category_id": 9,
              "score": 0.91, "bbox": [x, y, w, h],
              "mask": {"format": "rle", "size": [H, W], "counts": "..."} } ]
```

popoe's pose service accepts a frame manifest together with detections and runs the pose pipeline. It does not import NIDS-Net.
