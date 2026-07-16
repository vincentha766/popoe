"""popoe.recipes — evaluated-best stage configurations, in one place.

``best_recipe`` wires the configuration that produced the strongest measured
numbers in the reproduction study (YCB-V full BOP AR 0.7668 / LM-O 0.6726 with
the official CNOS-FastSAM detections):

  * DINOv2 ViT-g intermediate layer (FoundPose depth ratio) + object crop;
  * two-scale GeDi (30% + 40% of diameter, 64-D geometric);
  * fused [vis|geo] at PCA-matched dims, visual-weight sweep at selection time;
  * 32x32 target sampling grid (the "formal" density; 16 is the fast preset);
  * Open3D feature-matching RANSAC + ICP, thresholds at 3% of object extent
    (metric space — equivalent to the canonical-space 0.03 used in the study);
  * ChampionScorer (icp * s_feat_1, size-aware for pooled confusable pairs);
  * label pooling for confusable same-shape pairs (YCB-V clamps 19/20).

Heavy models load lazily on first use; everything here is metric-space.
"""

from __future__ import annotations

import os

import numpy as np

# The YCB-V clamp pair — same shape, different size. See segmentor_detections.
YCBV_MERGE_LABELS = {19: [19, 20], 20: [19, 20]}

# Selection-time visual-weight menu (per-target argmax via ChampionScorer).
WEIGHTS = (1.0, 0.7, 0.5, 0.3, 0.2)

TAU_FRAC = 0.03          # RANSAC/ICP threshold as a fraction of object extent


def scale_vis(feats: np.ndarray, w: float) -> np.ndarray:
    """Rescale the visual half of [vis | geo] fused features extracted at w=1.
    Extracting once and rescaling reproduces any weight exactly."""
    vd = feats.shape[1] // 2
    out = feats.astype(np.float64).copy()
    out[:, :vd] *= w
    return out


def best_encoders(device: str = "cuda", target_grid: int = 32,
                  render_backend: str = "auto"):
    """Shared-model query/target encoders at the formal configuration.
    Returns (query_encoder, target_encoder). GPU required.

    Extraction is PINNED to vis_weight=1.0: `scale_vis` and the selection-time
    weight sweep are both specified against w=1 features ("extracted at w=1",
    above), and ChampionScorer's s_feat_1 is documented as a w=1 re-score. The
    pin makes that true regardless of POPOE_VIS_WEIGHT — previously the env
    default (0.5) leaked in, so every "w" in the sweep and the "w=1" re-score
    actually ran at half the advertised visual weight.

    `render_backend='nvdiffrast'` refuses to run on a box without the GPU
    rasteriser rather than silently producing CPU-ray-cast features, which are
    NOT the same features (see QueryFeatureExtractor). The evaluated numbers
    were produced on nvdiffrast."""
    os.environ.setdefault("POPOE_TARGET_GRID", str(target_grid))
    from popoe.adapters import make_freeze_encoders
    from popoe.feature_extractor import (
        QueryFeatureExtractor, TargetFeatureExtractor, load_dinov2, load_gedi)
    dino = load_dinov2(device)
    gedi = load_gedi(device)
    qx = QueryFeatureExtractor(device, dino=dino, gedi=gedi,
                               render_backend=render_backend)
    tx = TargetFeatureExtractor(device, dino=dino, gedi=gedi)
    qx.fusion.vis_weight = 1.0   # pinned; make_freeze_encoders shares qx.fusion
    return make_freeze_encoders(qx, tx)


def best_segmentor(detections_json: str | None = None, topk: int = 2,
                   merge_labels: dict | None = None, sources=None):
    """Detections segmentor over one file (`detections_json`) or a union of
    NAMED backends (`sources` — dict {name: path}, DetectionSource/(name, path)
    list, or 'name=path' strings; see BOPDetectionsSegmentor). Exactly one of
    the two must be given."""
    if (detections_json is None) == (sources is None):
        raise ValueError("pass exactly one of detections_json or sources")
    from popoe.segmentor_detections import BOPDetectionsSegmentor
    if sources is not None:
        return BOPDetectionsSegmentor(sources=sources, topk=topk,
                                      merge_labels=merge_labels)
    return BOPDetectionsSegmentor(detections_json, topk=topk,
                                  merge_labels=merge_labels)


def stages_for_object(extent_m: float, size_aware: bool = False,
                      n_ransac: int = 10000, score_coarse: bool = False):
    """Per-object solver/refiner/scorer with thresholds scaled to the object.
    ``extent_m``: max bounding-box side of the sampled query cloud (metres).

    ``score_coarse=True`` records the paper's S_coarse (pre-ICP feature score)
    into the scorer breakdown as a DIAGNOSTIC — the final score is unchanged, so
    the evaluated config is byte-identical when it is off. It wires the coarse
    pose through: ICPRefiner(keep_coarse=True) + ChampionScorer(compute_s_coarse
    =True)."""
    from popoe.adapters import ICPRefiner
    from popoe.scoring import ChampionScorer
    from popoe.solvers import Open3DFeatureRansacSolver
    tau = TAU_FRAC * extent_m
    solver = Open3DFeatureRansacSolver(tau_inlier=tau, max_iteration=n_ransac)
    refiner = ICPRefiner(tau_icp=tau, keep_coarse=score_coarse)
    scorer = ChampionScorer(tau_inlier_frac=TAU_FRAC, size_aware=size_aware,
                            compute_s_coarse=score_coarse)
    return solver, refiner, scorer
