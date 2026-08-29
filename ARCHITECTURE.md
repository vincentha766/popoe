# Architecture

popoe factors a 6-DoF pose pipeline into **swappable stages**, each
a `typing.Protocol` in [src/popoe/interfaces.py](src/popoe/interfaces.py). An
implementation only needs matching method signatures — no base class, no
registration — so stages stay decoupled and any one can be re-implemented alone.

Measured claims, run artefacts, and withdrawn tables live in
[REPRODUCTION.md](REPRODUCTION.md). Incident write-ups that taught a rule live
in [ISSUES.md](ISSUES.md). This file is the seams and the invariants.

## Stages

```
ObjectModel (CAD) ─┬─ QueryEncoder ──────────── q, CanonFrame ─┐
                   ├─ Segmentor ─ Detection ─┐                 │
Scene (RGB-D, K) ──┴─────────────────────────┴─ TargetEncoder ─┴─ PoseSolver ─ PoseRefiner* ─ PoseScorer ─ Selector ─ (R, t)
```

| Stage | Protocol | Reference implementation |
|-------|----------|--------------------------|
| Segmentation | `Segmentor` | `segmentor_detections.BOPDetectionsSegmentor` (evaluated) — more in [§Segmentation backends](#segmentation-backends) |
| Query features | `QueryEncoder` | `freeze.adapters.FreeZeQueryEncoder` (DINOv2 visual + `PointDescriptor` geometric branch) |
| Target features | `TargetEncoder` | `freeze.adapters.FreeZeTargetEncoder` |
| Geometric descriptors | `PointDescriptor` | `freeze.feature_extractor.load_geometric_descriptor` dispatches on `POPOE_GEOM_BACKBONE`: `load_gedi` (default); `descriptors.FPFHDescriptor`; dGeDi via `POPOE_GEOM_BACKBONE` |
| Fusion | `FeatureFusion` | `freeze.fusion.DinoGeDiFusion` |
| Pose solve | `PoseSolver` | `solvers.Open3DFeatureRansacSolver` (default) — 3 more in [§Pluggability](#pluggability-proven--the-posesolver-stage) |
| External coarse pose | `CoarseEstimator` | `segmentor_sam6d.SAM6DPemResultsCoarseEstimator` over already-written PEM results |
| Refine | `PoseRefiner` | `adapters.ICPRefiner` |
| Score | `PoseScorer` | `freeze.adapters.FreeZeScorer`; `scoring.ChampionScorer` (evaluated) |
| Render re-rank (opt.) | `PoseRefiner` chain | `render_rerank.RenderAppearanceReranker` (knife-4 SAR-style; `--render-rerank`) |
| Select | `Selector` | `adapters.BestScoreSelector` |
| Metrics | `Metric` | `metrics.vsd`, `metrics.ar` |

The reference control flow is `interfaces.Pipeline.run`.

## Cross-cutting data (conventions live in one place)

`Scene`, `ObjectModel`, and `CanonFrame` are built once and threaded through
the pipeline, carrying the conventions that would otherwise be re-derived per
module and drift. `FrameManifest` sits one step earlier as the file-level input
boundary:

- **Units** — mesh vertices in mm; depth-unprojected points and output `t` in
  **metres** (BOP CSVs convert back to mm at the edge).
- **Frame I/O** — `FrameManifest` is the file boundary for RGB/depth/K plus an
  optional detections JSON. Detections remain 2D masks/scores; depth stays with
  the frame loader and is converted to metres before `Scene` is constructed.
- **Canonicalisation** — `CanonFrame` encodes `pts_canon = (pts - center) * scale`
  with `center = 0` and `scale = 1 / max_extent` of the query sampled cloud (NOT
  the BOP diameter): GeDi was trained at ~1 m, so the object is rescaled to ~1 m.
  The frame is an OUTPUT of query encoding (it depends on the sampled points) and
  is reused on the target side.

## Design rationale (why these seams)

- **Fusion is its own component.** `[w·L2(PCA(f_vis)), L2(f_geo)]` used to be
  copy-pasted inside both encoders; extracting `DinoGeDiFusion` makes the whole
  pure-geometric / pure-visual / fused ablation a one-liner
  (`DinoGeDiFusion(vis_weight=0.0 | 1.0 | ...)`) and lets query & target **share
  one fusion instance**, so the visual PCA fit on the query side is transparently
  reused on the target side.
- **The geometric branch is a component too.** The FreeZe recipe uses GeDi, but
  the encoders call only `PointDescriptor.compute(pts, pcd)`. That makes FPFH a
  proper hand-crafted control and dGeDi a fast learned control without changing
  query/target encoding, fusion, solving, or scoring. Descriptor radii are in
  canonical units (object extent ~= 1.0), so GeDi's `r_lrf` and FPFH's radii are
  comparable. Role-aware descriptors go through `descriptors.describe(...,
  role="query"|"target")`; role-blind descriptors keep the two-argument form.
- **Scoring is a stage, not baked into the refiner.** `PoseScorer` owns the whole
  feature-scoring concern: the fine re-score at the refined pose, and how the
  evidence combines. The combination rule belongs to the implementation, not to
  the pipeline — `FreeZeScorer` reproduces the paper's `s_coarse·s_fine·s_icp`,
  while the evaluated `ChampionScorer` uses `s_icp · s_feat_1 · metric_fit`, with
  `s_coarse` an opt-in per-dataset factor. `ICPRefiner` only moves geometry and
  reports `s_icp`. So a new solver or refiner never re-implements the scoring
  rule. (The RANSAC-internal inlier score stays inside the solver — that is
  hypothesis ranking, not final scoring.)
- **A solver only PROPOSES; the scorer DISPOSES.** See
  [§Pluggability](#pluggability-proven--the-posesolver-stage).
- **No stage hides a fallback.** A stage whose backend is missing raises
  `interfaces.BackendUnavailable` — it never quietly substitutes a weaker method.
  See [§Availability](#the-availability-contract-no-hidden-fallbacks).

## The availability contract (no hidden fallbacks)

Two different methods behind one name is a bug, not a convenience.

- an implementation raises `BackendUnavailable` (`SegmentorUnavailable`,
  `RendererUnavailable`) when a package / checkpoint / device is missing;
- a **runtime** failure (CUDA OOM, corrupt mesh) propagates — "the fallback
  handled it" is how real bugs get buried;
- substitution is the **caller's** policy: compose
  `segmentor.FirstAvailableSegmentor([...])`, then read `chain.last_used` and
  `Detection.source` to see what ran;
- anything that selects a method (`render_backend`, the segmentor's `source`)
  is part of the stage config and belongs **in the cache key**.

A silent substitution makes results unattributable (logs still name the method
you asked for) and poisons the config-addressed cache (the key fingerprints the
config, not the method that actually ran). The incidents that taught this —
CNOS swapping to a sliding window then depth blobs, the renderer reusing a
CPU-built cache on GPU — are in [ISSUES.md](ISSUES.md).

## Pluggability proven — the PoseSolver stage

Four `PoseSolver` implementations run through the identical
encoders→refiner→scorer→selector chain. A solver may return several hypotheses
and leave the choice to the scorer and selector, so "geometry proposes, features
dispose" is reachable as pure composition, with no new scoring code.

- `adapters.RansacSolver` — hand-rolled feature-aware RANSAC.
- `solvers.Open3DFeatureRansacSolver` — Open3D's C++ correspondence RANSAC.
  `n_restarts>1` emits several geometrically-ranked hypotheses; the feature-aware
  scorer re-ranks the survivors (the A layer).
- `solvers.GPURansacSolver` — batched RANSAC (vectorised triplet sampling +
  batched Kabsch/SVD; CPU or CUDA) with a selectable `fitness`, so feature
  agreement can sit **inside** hypothesis selection (the B layer), not only after
  it. `"geometric"` ranks by inlier count. `"feature"` uses the paper's Eq.5
  `Σ_inlier cos(f_q,f_t) / |P_T|`: the denominator is the **fixed** sparse-target
  count, never the inlier count (mean cosine lets a few high-similarity spurious
  correspondences beat many true ones). Features are the **w=1** canonical space.
- `solvers.TeaserSolver` — TEASER++ (Yang, Shi & Carlone, T-RO 2021). Prunes the
  correspondence pool with a pairwise TIM max-clique and solves rotation by
  GNC-TLS; deterministic, no RNG. Correspondences come from the same Eq.3
  per-target top-k cosine NN pool as `GPURansacSolver` (w=1 features);
  `tau_inlier` doubles as TEASER's noise bound. The import is deferred to
  `.solve`, so construction is dep-light.

The A/B that ranks the first three against each other is
[examples/solver_swap_demo.py](examples/solver_swap_demo.py); the numbers live
only in [REPRODUCTION.md](REPRODUCTION.md#solver-ab-ledger-2026-07-26) and are
not a performance claim for popoe. Default solver stays `o3d`, so the evaluated
mainline is unperturbed; the others are independent configurations.

## Segmentation backends

Every entry satisfies the same `Segmentor` protocol and stamps its origin into
`Detection.source`. **File** backends replay an artefact another process wrote;
**live** backends run the models themselves. How-to, producer pins, and naming
politics for each source live in [CNOS.md](CNOS.md), [MUSE.md](MUSE.md),
[NIDS_NET.md](NIDS_NET.md), [SAM6D.md](SAM6D.md).

| Implementation | `source` | Kind |
|----------------|----------|------|
| `segmentor_detections.BOPDetectionsSegmentor` | `bop-detections`, or per-source in a union | file — **evaluated**; one JSON or a named-source union |
| `segmentor_cnos_official.CNOSDetectionsSegmentor` | `cnos` | file — official CNOS / CNOS-FastSAM producer, or public BOP default detections |
| `segmentor_sam6d.SAM6DIsmDetectionsSegmentor` | `sam6d` | file — SAM-6D ISM artefacts |
| `segmentor_nids.NIDSNetDetectionsSegmentor` | `nids` | file — NIDS-Net artefacts |
| `segmentor_muse.MuseDetectionsSegmentor` | `muse-repro` | file — replay of dumped MUSE masks |
| `segmentor_muse.MuseSegmentor` | `muse-repro` | live — GroundingDINO→SAM2→DINOv2; also its own producer |
| `segmentor_cnos_lab.CNOSLabSegmentor` | `cnos-lab` | live — depth-size-gated foreground-patch CNOS (formerly `cnos-v3`) |
| `segmentor.SAMSegmentor` | `sam2-amg` | live — SAM2.1 automatic mask generator, class-agnostic |
| `segmentor.DepthSegmentor` | `depth-cc` | live — depth connected components; no model, no GPU |
| `segmentor.FirstAvailableSegmentor` | delegates | explicit fallback chain — see the availability contract |

CNOS-FastSAM, SAM-6D ISM and NIDS-Net all publish the same artefact — a
detections JSON — so they are not separate pose-backend code paths, only
different named producers. Official checkouts are pinned under `external/` for
source provenance but still run in separate environments; popoe-side adapters
consume the files. `segmentor_detections.DetectionSource` `(name, path)` is the
config handle: select a backend by name and compose several into one
`BOPDetectionsSegmentor`.

`topk` is per `(source, label)`, so a top-M union keeps M candidates **per
source**. The union across sources is **unfiltered** (FreeZe's "top-M union
without filtering"): `iou_dedupe` is scoped per source, two backends proposing
the same region both survive, and the feature-aware scorer disposes. Official
source names (`cnos`, `muse`) are reserved for official artefacts; lab /
reimplementation outputs use `cnos-lab` / `muse-repro`.

MUSE occupies both forms at once (`MuseSegmentor` live, `MuseDetectionsSegmentor`
file replay) because there is no public producer to adapt. Two design points
follow from scoring classes *jointly* rather than independently, and neither is
optional:

- **Classes are registered up front.** `Segmentor.segment` is a per-object
  contract, but MUSE's relative score is a softmax across all candidate classes.
  So the segmentor computes a `(proposal x class)` score matrix once and serves
  one column per call. A single registered class makes that score the constant 1
  and silently reduces the method to `beta * S_abs` — the availability
  contract's "two methods behind one name" in miniature, so it must be asked for
  explicitly (`allow_single_class=True`).
- **Proposals are per-frame.** Grounding DINO + SAM2 would otherwise re-run for
  every object in one image. Results are memoised by frame CONTENT, never by
  `scene_id`/`im_id` (real captures leave those at -1) — the same
  content-addressing invariant the cache follows.

SAM-6D's ISM half is a detections producer like the others. Its PEM half is an
external **full pose** producer, so it is not a `Segmentor` at all:
`segmentor_sam6d.SAM6DPemResultsCoarseEstimator` adapts PEM outputs to
`PoseHypothesis` through the separate `CoarseEstimator` contract.

## Assembly: sharing the heavy models (`popoe.assembly`)

Three components load their own DINOv2 ViT-g, and SAM2 is loaded independently
by the MUSE refiner and the AMG proposers. Composed as separate processes the
live ensemble plus pose duplicates those weights. Measured footprint:
[DEMO_SINGLE_GPU.md](DEMO_SINGLE_GPU.md).

`assembly.ModelPool` holds one lazily-loaded copy of each shared model; every
model-owning component grew a matching injection parameter (`model=` on the
DINOv2 pair, `sam_model=` on the SAM2 trio) that bypasses its own loader. The
`*_from_pool` builders wire the existing segmentors and pose encoders around
one pool. Two scoped departures from the usual rules: pool wiring loads the
*pooled* models at build time, not first use (a resident service wants startup
failures at startup — though unpooled components like MUSE's Grounding DINO
still lazy-load on the first frame); and component `config()` identity still
describes models by name — truthful because the pool is the single place a name
resolves to weights.

The service shell that would sit on top (HTTP, cameras, supervision) stays
outside popoe (AGENTS.md boundary); this module only guarantees the library
composes into one process without duplicate weights.

## Verification

- **Adapter fidelity** — [examples/pipeline_selfcheck.py](examples/pipeline_selfcheck.py):
  the adapter chain reproduces the inline `FreeZeV2.estimate_pose` body
  (`examples/freezev2_monolith.py`) to ~1e-15
  on identical arrays (fixed RANSAC seed + deterministic ICP).
- **Fusion byte-identity & Protocol wiring** — [tests/](tests/), GPU-free
  (numpy + scikit-learn), run with `pytest`.

Runner-level BOP invariants (one query encode per object, w=1 extraction,
`inst_count` row-count resume, `--cand-csv`, confusable-object arbitration)
live on `examples/bop_eval.py`, not here.

## Stage caching (config-addressed)

Because stages are separable, their outputs are cacheable — `popoe.cache`
keys every stage output by a fingerprint of (stage config, input CONTENT,
and the keys of any upstream fits it depends on). Same configuration →
automatic reuse; changing a knob invalidates exactly the entries it should.

Three invariants, all learned from real incidents (see ISSUES.md):

1. **Fitted state is part of the key.** The target-feature key includes the
   QUERY key, because the query's fitted visual PCA defines the basis the
   target features live in. (Violation: silent cross-run basis mismatch;
   texture-reliant objects crater.)
2. **Content addressing, not positional indices.** A mask's identity is a
   hash of its pixels, never its index in a detection list. (Violation:
   pooling reorders the list and a *different* mask's features load.)
3. **Every feature-changing knob is config.** The key records the effective
   target grid, DINO layer, crop/fill/canon settings, render backend, geometric
   backbone, dGeDi mode, GeDi path, and, when FPFH is active, the FPFH radii,
   voxel, normal and orientation settings. Missing one turns a sweep into a
   stale-feature replay.
