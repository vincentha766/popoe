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

Lab-only (opt-in, **not** the headline path): mask-stage
``size_select`` via :func:`ycbv_lab_segmentor` / ``best_segmentor(...,
size_select=...)`` / ``bop_eval --size-select``. Formal defaults keep
``size_select=None``.

Heavy models load lazily on first use; everything here is metric-space.
"""

from __future__ import annotations

import os

import numpy as np

# The YCB-V clamp pair — same shape, different size. See segmentor_detections.
YCBV_MERGE_LABELS = {19: [19, 20], 20: [19, 20]}

# BOP models_info diameters (metres) for the clamp pair. Used by optional
# mask-stage size_select (nearest / soft); registration still uses
# ChampionScorer(size_aware=True) when the object is in YCBV_MERGE_LABELS.
YCBV_CLAMP_DIAMETERS_M = {
    19: 0.174746,   # large_clamp
    20: 0.217094,   # extra_large_clamp
}

# Selection-time visual-weight menu (per-target argmax via ChampionScorer).
WEIGHTS = (1.0, 0.7, 0.5, 0.3, 0.2)

TAU_FRAC = 0.03          # RANSAC/ICP threshold as a fraction of object extent


def scale_vis(feats: np.ndarray, w: float,
              vis_dim: int | None = None) -> np.ndarray:
    """Rescale the VISUAL part of [vis | geo] fused features extracted at w=1.
    Extracting once and rescaling reproduces any weight exactly.

    The split is `vis_dim` wide, NOT necessarily half. `feats.shape[1] // 2` is
    right only while the visual branch is geo-matched — the default, and what
    every published number ran under (64-D GeDi -> 64-D visual -> 128-D fused).
    Set `POPOE_VIS_DIM` to anything else and the halves stop being equal: 1536-D
    visual against 64-D GeDi fuses to 1600-D, where `// 2` puts the boundary at
    800 and a weight sweep would scale 800 of the 1536 visual channels and leave
    the other 736 at w=1 — a sweep silently running a different weighting than
    its label says, the same shape as the "w=1 was never w=1" defect
    (ISSUES.md 2026-07-14).

    `vis_dim=None` keeps the historical equal-halves answer, so no existing
    caller changes behaviour. The split is NOT inferred from `POPOE_VIS_DIM`:
    that env var is only the fusion's *default*, and a caller that passes
    `DinoGeDiFusion(vis_dim=...)` or a pre-fit `pca_vis` overrides it — the real
    width is then `pca_vis.n_components`, and reading the env would give a
    boundary the features never had. (Measured: with a 64-component PCA
    supplied and `POPOE_VIS_DIM=1536` in the environment, the env answer tries
    to split a 128-D vector at 1536.)

    So the caller that knows must say so. `bop_eval` does, from
    `enc_cfg['vis_dim']` — the same value its encoder cache key records, which
    is what guarantees the split matches the features it is applied to:
    a different setting is a different key, hence a different entry.

    Since PR #12 the fused visual width IS `vis_dim` on every surviving path
    (PCA projects to `n_components=vis_dim`; the identity path requires
    `n_vis == vis_dim`), so a caller can always determine it — there is no
    third case where the width is whatever a raw slice produced.
    """
    if vis_dim is None:
        vis_dim = feats.shape[1] // 2
    if not 0 < vis_dim < feats.shape[1]:
        raise ValueError(
            f"vis_dim {vis_dim} is not a valid split of {feats.shape[1]}-D "
            f"fused features — the visual part must be a proper prefix. "
            f"Scaling with a wrong boundary silently reweights the wrong "
            f"channels, so this refuses rather than guesses.")
    out = feats.astype(np.float64).copy()
    out[:, :vis_dim] *= w
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
    from popoe.freeze.adapters import make_freeze_encoders
    from popoe.freeze.feature_extractor import (
        QueryFeatureExtractor, TargetFeatureExtractor, load_dinov2, load_gedi)
    dino = load_dinov2(device)
    gedi = load_gedi(device)
    qx = QueryFeatureExtractor(device, dino=dino, gedi=gedi,
                               render_backend=render_backend)
    tx = TargetFeatureExtractor(device, dino=dino, gedi=gedi)
    qx.fusion.vis_weight = 1.0   # pinned; make_freeze_encoders shares qx.fusion
    return make_freeze_encoders(qx, tx)


def best_segmentor(detections_json: str | None = None, topk: int = 2,
                   merge_labels: dict | None = None, sources=None,
                   size_select: str | None = None,
                   confusable_diameters: dict | None = None,
                   size_select_fallback: bool = True):
    """Detections segmentor over one file (`detections_json`) or a union of
    NAMED backends (`sources` — dict {name: path}, DetectionSource/(name, path)
    list, or 'name=path' strings; see BOPDetectionsSegmentor). Exactly one of
    the two must be given.

    ``size_select`` is opt-in mask-stage size arbitration for confusable pairs
    (``None`` default = formal BOP headline path). See
    :func:`ycbv_lab_segmentor` for the lab clamp recipe.
    """
    if (detections_json is None) == (sources is None):
        raise ValueError("pass exactly one of detections_json or sources")
    from popoe.segmentor_detections import BOPDetectionsSegmentor
    kw = dict(
        topk=topk,
        merge_labels=merge_labels,
        size_select=size_select,
        confusable_diameters=confusable_diameters,
        size_select_fallback=size_select_fallback,
    )
    if sources is not None:
        return BOPDetectionsSegmentor(sources=sources, **kw)
    return BOPDetectionsSegmentor(detections_json, **kw)


def ycbv_lab_segmentor(detections_json: str | None = None, topk: int = 2,
                       sources=None, size_select: str = "nearest",
                       size_select_fallback: bool = True):
    """YCB-V *lab / confusable-clamp* segmentor (not the formal headline path).

    Combines formal merge labels with mask-stage nearest/soft size select.
    Does **not** change ``best_segmentor`` defaults or REPRODUCTION headline
    numbers — call this (or pass ``size_select=`` explicitly) for lab runs.
    """
    if size_select not in ("soft", "nearest"):
        raise ValueError(
            f"ycbv_lab_segmentor size_select must be 'soft' or 'nearest', "
            f"got {size_select!r}"
        )
    return best_segmentor(
        detections_json=detections_json,
        sources=sources,
        topk=topk,
        merge_labels=YCBV_MERGE_LABELS,
        size_select=size_select,
        confusable_diameters=dict(YCBV_CLAMP_DIAMETERS_M),
        size_select_fallback=size_select_fallback,
    )


SOLVERS = ("o3d", "gpu", "gpu-feat", "teaser")   # o3d is the evaluated mainline default


def _build_solver(name: str, tau: float, n_ransac: int, seed: int | None = None):
    """o3d (default, mainline) | gpu (ported RANSAC, geometric fitness) |
    gpu-feat (gpu with the Eq.5 feature-aware fitness — the B layer) |
    teaser (TEASER++ certifiable registration; deterministic, so n_ransac
    does not apply — needs teaserpp_python, see popoe.solvers.teaser).

    ``seed`` reaches the two stochastic solvers only:

      * o3d — ``seed=None`` (default) leaves Open3D's global RNG unseeded,
        which is the historical behaviour every evaluated number was produced
        under. Passing a seed makes the run reproducible and IS a different
        configuration; REPRODUCTION.md's acceptance rule ("tighten to
        bit-identical if the run is seeded") is what it exists for.
      * gpu / gpu-feat — already deterministic (their own default is 42); a
        seed here overrides that, ``None`` keeps it.
      * teaser — no RNG at all, so the seed is not applicable and is ignored.
    """
    if name == "o3d":
        from popoe.solvers import Open3DFeatureRansacSolver
        return Open3DFeatureRansacSolver(tau_inlier=tau, max_iteration=n_ransac,
                                         seed=seed)
    if name in ("gpu", "gpu-feat"):
        from popoe.solvers import GPURansacSolver
        kw = {} if seed is None else {"seed": seed}   # else keep its own default
        return GPURansacSolver(tau_inlier=tau, iters=n_ransac,
                               fitness="feature" if name == "gpu-feat" else "geometric",
                               **kw)
    if name == "teaser":
        from popoe.solvers import TeaserSolver
        return TeaserSolver(tau_inlier=tau)
    raise ValueError(f"solver must be one of {SOLVERS}, got {name!r}")


def solver_provenance(name: str, seed: int | None) -> str:
    """One log line describing the run's solver determinism.

    Reads the EFFECTIVE seed off a throwaway solver rather than restating the
    defaults, so the answer cannot drift from `_build_solver`. Construction is
    dep-light (torch/open3d/teaserpp all import inside `.solve`).

    Only o3d is genuinely unseeded by default; the gpu solvers carry their own
    deterministic default and teaser has no RNG at all. Saying "UNSEEDED" for
    all of them would put a false claim in the provenance of a cited run.
    """
    if name == "teaser":
        return f"solver={name} seed=n/a (deterministic — TEASER++ has no RNG)"
    effective = getattr(_build_solver(name, tau=1.0, n_ransac=1, seed=seed),
                        "seed", None)
    if effective is None:
        return (f"solver={name} seed=None (UNSEEDED — Open3D's global RNG is "
                f"never seeded, so poses vary run-to-run)")
    return f"solver={name} seed={effective} (deterministic)"


def stages_for_object(extent_m: float, size_aware: bool = False,
                      n_ransac: int = 10000, score_coarse: bool = False,
                      use_s_coarse: bool = False, solver: str = "o3d",
                      seed: int | None = None,
                      tau_basis_m: float | None = None):
    """Per-object solver/refiner/scorer with thresholds scaled to the object.
    ``extent_m``: max bounding-box side of the sampled query cloud (metres).

    S_coarse (pre-ICP feature score, canonical w=1) is wired here:
      * ``score_coarse=True`` RECORDS it into the breakdown (diagnostic; score
        unchanged).
      * ``use_s_coarse=True`` USES it as an arbitration factor (score *=
        max(s_coarse, 0)) — a per-DATASET switch (helps YCB-V, hurts LM-O).
    Either wires the coarse pose through (ICPRefiner(keep_coarse=True)); with
    both off the config is byte-identical.

    ``seed`` is forwarded to the solver (see :func:`_build_solver`); ``None``
    is the historical, unseeded mainline.

    ``tau_basis_m``: what the 3% in :data:`TAU_FRAC` is 3% OF, in metres.
    ``None`` (default, historical) keeps ``extent_m`` — the largest side of the
    sampled query cloud's axis-aligned bounding box. FreeZeV2 Sec. V-A says
    "Thresholds tau_inlier and tau_ICP are set to 3% of the object's diameter",
    and a bounding-box side is never larger than the diameter (on LM-O it runs
    2-35% short), so every threshold derived here is systematically TIGHTER than
    the paper's. Pass the BOP ``models_info`` diameter to follow the paper. The
    one basis feeds all three thresholds it decides — RANSAC's tau_inlier, ICP's
    tau_ICP, and the feature score's inlier radius (Eq. 4 defines a single
    tau_inlier that Eq. 5 reuses), so they cannot drift apart."""
    from popoe.adapters import ICPRefiner
    from popoe.scoring import ChampionScorer
    tau = TAU_FRAC * (extent_m if tau_basis_m is None else tau_basis_m)
    solver = _build_solver(solver, tau, n_ransac, seed=seed)
    refiner = ICPRefiner(tau_icp=tau, keep_coarse=score_coarse or use_s_coarse)
    # use_s_coarse implies s_coarse IS computed and emitted, so compute_s_coarse
    # reflects reality (an inspector reading the flag sees the truth).
    scorer = ChampionScorer(tau_inlier_frac=TAU_FRAC, size_aware=size_aware,
                            compute_s_coarse=score_coarse or use_s_coarse,
                            use_s_coarse=use_s_coarse,
                            # None on the historical path: the scorer then
                            # recomputes TAU_FRAC * extent itself, identically.
                            tau_abs=None if tau_basis_m is None else tau)
    return solver, refiner, scorer
