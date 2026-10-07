# SAM-6D

SAM-6D has two halves, and they enter popoe through different contracts.
ISM is a detection producer like CNOS and NIDS-Net.
PEM is a full-pose producer, so it is not a `Segmentor` at all — `segmentor_sam6d.SAM6DPemResultsCoarseEstimator` adapts its output to `PoseHypothesis` through the `CoarseEstimator` contract.

The producer boundary, the shared checks, and `BOPDetectionsSegmentor` usage are described in [README.md](README.md).

```text
SAM-6D ISM env/service
  RGB/templates -> 2D detections JSON       -> source="sam6d"

SAM-6D PEM env/service
  RGB-D + CAD + detections -> pose CSV/JSON -> source="sam6d-pem"
```

PEM translations are in millimetres, and popoe converts them to metres when constructing a `PoseHypothesis`. Confirm the unit before relying on the default `translation_scale=0.001`.

## Upstream

The official code is pinned as a submodule:

```bash
git submodule update --init --recursive external/SAM-6D
git -C external/SAM-6D rev-parse HEAD
```

The official stack has substantially more dependencies than popoe and runs in its own environment:

```bash
export POPOE_SAM6D_PATH=/path/to/SAM-6D      # defaults to external/SAM-6D
export POPOE_SAM6D_PYTHON=/path/to/envs/sam6d/bin/python
```

## Instance segmentation (ISM)

Build the official command from popoe:

```bash
popoe-sam6d ism-command --dataset lmo --model fastsam --gpu 0
```

It prints the equivalent of:

```bash
cd external/SAM-6D/SAM-6D/Instance_Segmentation_Model
CUDA_VISIBLE_DEVICES=0 python run_inference.py dataset_name=lmo model=ISM_fastsam
```

The resulting `result_<dataset>.json` is an ordinary detections file. Officially published files, with hashes: [`data/detections/sam6d/PROVENANCE.md`](../../data/detections/sam6d/PROVENANCE.md).

```python
from popoe.segmentor_detections import BOPDetectionsSegmentor

seg = BOPDetectionsSegmentor(
    "external/SAM-6D/SAM-6D/Instance_Segmentation_Model/log/sam/result_lmo.json",
    source="sam6d", topk=2,
)
```

## Pose estimation (PEM)

Build the official command:

```bash
popoe-sam6d pem-command \
  --dataset lmo \
  --checkpoint-path checkpoints/sam-6d-pem-base.pth \
  --gpus 0 --view 42
```

The official script reads its detection paths from `test_bop.py`, and custom segmentation inputs require editing the `detetion_paths` mapping in that file. Keep such edits in the SAM-6D checkout, not in popoe.

PEM writes BOP pose CSV rows. Load them as coarse hypotheses and compose them as a `DirectPoseMethod`, the estimator graph, rather than as a correspondence `Pipeline`:

```python
from popoe import DirectPoseMethod
from popoe.adapters import BestScoreSelector, ICPRefiner
from popoe.segmentor_sam6d import SAM6DPemResultsCoarseEstimator

est = SAM6DPemResultsCoarseEstimator(
    "external/SAM-6D/SAM-6D/Pose_Estimation_Model/log/"
    "pose_estimation_model_base_id0/lmo_eval_iter000000/result_lmo.csv",
    topk=1,
)
hyps = est.estimate(scene, obj)      # list[PoseHypothesis], t in metres

method = DirectPoseMethod(estimator=est, selector=BestScoreSelector())
hyp = method.run(scene, obj)
```

Geometry-only ICP is optional. The default clouds are `popoe.adapters.icp_clouds` (CAD + depth, in metres), and this graph has no query or target features:

```python
method = DirectPoseMethod(
    estimator=est, selector=BestScoreSelector(),
    geometric_refiners=[ICPRefiner(tau_icp=0.03)],
)
```

## Real scene mode

The official custom demo can write `OUTPUT_DIR/sam6d_results/detection_ism.json` and `detection_pem.json`.

Consume `detection_ism.json` with `SAM6DIsmDetectionsSegmentor` when records have `scene_id`, `image_id`, `category_id`, `score` and a mask field. Consume `detection_pem.json` with `SAM6DPemResultsCoarseEstimator` when each record has `R` and `t`.

Custom output frequently omits ids or leaves them at a placeholder (`-1` or `0`). The `scene_id`, `image_id` and `obj_id` arguments assign ids to such records; they do not filter. A record that already carries a different, non-placeholder id raises an error rather than being overwritten:

```python
est = SAM6DPemResultsCoarseEstimator(
    "captures/frame_000042/sam6d_results/detection_pem.json",
    scene_id=0, image_id=42, obj_id=9,
)
```
