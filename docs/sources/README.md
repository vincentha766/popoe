# Detection and pose sources

Four external methods feed popoe. Each runs in its own environment, writes a file, and popoe consumes that file. popoe does not import any of them.

| Source tag | Producer | Artefact | How-to |
|---|---|---|---|
| `cnos` | official CNOS / CNOS-FastSAM | 2D detections JSON | [cnos.md](cnos.md) |
| `cnos-lab` | popoe's local CNOS recipe | 2D detections, live | [cnos.md](cnos.md#local-cnos-lab) |
| `sam6d` | SAM-6D ISM | 2D detections JSON | [sam6d.md](sam6d.md) |
| `sam6d-pem` | SAM-6D PEM | full pose CSV / JSON | [sam6d.md](sam6d.md#pose-estimation-pem) |
| `nids` | NIDS-Net | 2D detections JSON | [nids-net.md](nids-net.md) |
| `muse` | MUSE authors' BOP submissions | 2D detections JSON | [muse.md](muse.md) |
| `muse-repro` | popoe's MUSE reimplementation | 2D detections, live + file | [muse.md](muse.md) |

## The producer boundary

Every detection producer has the same shape:

```text
producer env/service
  RGB (+ templates) -> 2D detections JSON

popoe env/service
  RGB-D frame manifest + detections JSON -> 6D pose
```

A detections JSON carries only 2D information: `scene_id`, `image_id`, `category_id`, `score`, `bbox`, and `segmentation` (real captures may use `mask` or `mask_path` instead). Depth never travels in it. Depth stays in the frame manifest and is loaded into `Scene.depth` in metres.

SAM-6D's PEM half is the exception: it produces full poses rather than detections, with `t` in millimetres, which popoe converts to metres when building a `PoseHypothesis`.

Producer dependencies — Hydra, Grounding DINO, SAM / FastSAM, Detectron2, version-pinned support packages — move much faster than the pose backend, so they are deliberately absent from popoe's `pyproject.toml`.
Give each producer its own conda or uv environment.
On a single-GPU workstation, run them serially: run the producer, let that process exit and release GPU memory, then run popoe.

## Source naming

Official tags (`cnos`, `sam6d`, `nids`, `muse`) are reserved for artefacts the original authors published. Reimplementations write their own tag (`cnos-lab`, `muse-repro`). Keeping them apart is what stops a reimplementation's number from being cited as the official method's.

## Consuming a file

Every file-backed detection source goes through one class, with the tag passed explicitly so provenance survives into scoring and logs:

```python
from popoe.segmentor_detections import BOPDetectionsSegmentor

# one file
seg = BOPDetectionsSegmentor(
    "data/detections/cnos/cnos-fastsam_ycbv-test.json", source="cnos", topk=2
)

# named-source union; topk is per (source, label)
seg = BOPDetectionsSegmentor(sources={
    "cnos":  "data/detections/cnos/cnos-fastsam_ycbv-test.json",
    "nids":  "data/detections/nids/nids_wa_sappe_ycbv.json",
    "sam6d": "data/detections/sam6d/sam6d_official_ycbv.json",
}, topk=2)
```

Checks that apply to every source:

- object IDs match the CAD set popoe consumes;
- `scene_id` / `image_id` match the frame manifest;
- masks are present, not just boxes (BOP method pages list detection-only batches next to segmentation batches — check the Task field);
- the submodule commit is recorded before reproducing results;
- the source tag is the right one for who produced the file.

File identity is pinned per directory: `data/detections/<source>/PROVENANCE.md` holds the download and SHA256s, and `python scripts/freeze_detections.py --check` verifies what is on disk.
