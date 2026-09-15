# Architecture

popoe factors 6-DoF pose into a **method** (`PoseMethod.run(scene, obj)`) and **optional stage** Protocols in [src/popoe/interfaces.py](src/popoe/interfaces.py). A method uses only the stages its graph needs. An implementation only needs matching method signatures — no base class, no registration.

## Stages

Library entry: `PoseMethod.run(scene, obj) → PoseHypothesis | None`.

After encode, both `Pipeline.run` and `examples/bop_eval.py` call `correspond_pair` (solve → refine* → score). The runner still owns disk cache, the visual-weight sweep, multi-instance NMS, and resume.

Correspondence graph (`Pipeline` / `CorrespondencePipeline`):

```
ObjectModel (CAD) ─┬─ QueryEncoder ──────────── q, CanonFrame ─┐
                   ├─ Segmentor ─ Detection ─┐                 │
Scene (RGB-D, K) ──┴─────────────────────────┴─ TargetEncoder ─┴─ PoseSolver ─ PoseRefiner* ─ PoseScorer ─ Selector ─ (R, t)
```

Estimator graph (`DirectPoseMethod`):

```
Scene, ObjectModel ─ (Segmentor?) ─ CoarseEstimator ─ GeometricRefiner* ─ Selector ─ (R, t)
```

| Stage | Protocol | Reference implementation |
|-------|----------|--------------------------|
| Segmentation | `Segmentor` | `segmentor_detections.BOPDetectionsSegmentor` (evaluated) — more in [§Segmentation backends](#segmentation-backends) |
| Query features | `FreeZeQueryEncoder` | `freeze.adapters.FreeZeQueryEncoder` (DINOv2 visual + `PointDescriptor` geometric branch) |
| Target features | `FreeZeTargetEncoder` | `freeze.adapters.FreeZeTargetEncoder` |
| Geometric descriptors | `PointDescriptor` | `freeze.feature_extractor.load_geometric_descriptor` dispatches on `POPOE_GEOM_BACKBONE`: `load_gedi` (default); `descriptors.FPFHDescriptor` |
| Fusion | class | `freeze.fusion.DinoGeDiFusion` |
| Pose solve | `PoseSolver` | `solvers.Open3DFeatureRansacSolver` (default) — also GPU RANSAC and TEASER++ |
| External coarse pose | class | `segmentor_sam6d.SAM6DPemResultsCoarseEstimator` over already-written PEM results |
| Refine (correspondence) | `PoseRefiner` | `adapters.ICPRefiner` (clouds from encoded features) |
| Refine (estimator) | `GeometricRefiner` | `adapters.ICPRefiner.refine_geometry`; default clouds: `adapters.icp_clouds` |
| Score | class | `scoring.ChampionScorer` |
| Render re-rank (opt.) | `PoseRefiner` chain | `render_rerank.RenderAppearanceReranker` (`--render-rerank`) |
| Select | function | `adapters.best_hyp` / `select_top_instances` |
| Metrics | scripts | `metrics.vsd`, `metrics.ar` |

The library entry is `PoseMethod.run`. `make_correspondence_pipeline` in `popoe.freeze.recipes` returns a `Pipeline` (correspondence graph); `DirectPoseMethod` is the estimator-graph implementation (for example SAM-6D PEM files). The evaluated BOP loop is `examples/bop_eval.py` (cache, weight sweep, multi-instance, resume). It builds a per-object `Pipeline` with `make_correspondence_pipeline`, then scores each encoded pair with `correspond_pair`. Default eval flags are the tuned Open3D identity, not a paper-faithful freeze — see [README.md](README.md#minimal-bop-eval).

## Cross-cutting data (conventions live in one place)

`Scene`, `ObjectModel`, and `CanonFrame` are built once and threaded through the pipeline, carrying the conventions that would otherwise be re-derived per module and drift. `FrameManifest` sits one step earlier as the file-level input boundary:

- **Units** — mesh vertices in mm; depth-unprojected points and output `t` in **metres** (BOP CSVs convert back to mm at the edge).
- **Frame I/O** — `FrameManifest` is the file boundary for RGB/depth/K plus an optional detections JSON. Detections remain 2D masks/scores; depth stays with the frame loader and is converted to metres before `Scene` is constructed.
- **Canonicalisation** — `CanonFrame` encodes `pts_canon = (pts - center) * scale` with `center = 0` and `scale = 1 / max_extent` of the query sampled cloud (not the BOP diameter): GeDi was trained at ~1 m, so the object is rescaled to ~1 m. The frame is an output of query encoding (it depends on the sampled points) and is reused on the target side.

## Design rationale (why these seams)

- **Fusion is its own component.** `[w·L2(PCA(f_vis)), L2(f_geo)]` used to be copy-pasted inside both encoders; extracting `DinoGeDiFusion` makes the whole pure-geometric / pure-visual / fused ablation a one-liner (`DinoGeDiFusion(vis_weight=0.0 | 1.0 | ...)`) and lets query and target **share one fusion instance**, so the visual PCA fit on the query side is reused on the target side.
- **The geometric branch is a component too.** The FreeZe recipe uses GeDi, but the encoders call only `PointDescriptor.compute(pts, pcd)`. That makes FPFH a proper hand-crafted control and dGeDi a fast learned control without changing query/target encoding, fusion, solving, or scoring. Descriptor radii are in canonical units (object extent ~= 1.0), so GeDi's `r_lrf` and FPFH's radii are comparable. Role-aware descriptors go through `descriptors.describe(..., role="query"|"target")`; role-blind descriptors keep the two-argument form.
- **Scoring is a stage, not baked into the refiner.** `PoseScorer` owns the whole feature-scoring concern: the fine re-score at the refined pose, and how the evidence combines. The combination rule belongs to the implementation, not to the pipeline — `FreeZeScorer` reproduces the paper's `s_coarse·s_fine·s_icp`, while the evaluated `ChampionScorer` uses `s_icp · s_feat_1 · metric_fit`, with `s_coarse` an opt-in per-dataset factor. `ICPRefiner` only moves geometry and reports `s_icp`. A new solver or refiner never re-implements the scoring rule. (The RANSAC-internal inlier score stays inside the solver — that is hypothesis ranking, not final scoring.)
- **A solver only proposes; the scorer disposes.** See [§Pluggability](#pluggability--the-posesolver-stage).
- **No stage hides a fallback.** A stage whose backend is missing raises `interfaces.BackendUnavailable` — it never quietly substitutes a weaker method. See [§Availability](#the-availability-contract-no-hidden-fallbacks).

## The availability contract (no hidden fallbacks)

Two different methods behind one name is a bug, not a convenience.

- an implementation raises `BackendUnavailable` (`SegmentorUnavailable`, `RendererUnavailable`) when a package / checkpoint / device is missing;
- a **runtime** failure (CUDA OOM, corrupt mesh) propagates;
- substitution is the **caller's** policy: compose an explicit caller chain, then read `Detection.source` to see what ran;
- anything that selects a method (`render_backend`, the segmentor's `source`) is part of the stage config and belongs **in the cache key**.

A silent substitution makes results unattributable (logs still name the method you asked for) and poisons the config-addressed cache (the key fingerprints the config, not the method that actually ran).

## Pluggability — the PoseSolver stage

Three `PoseSolver` implementations run through the identical encoders→refiner→scorer chain. A solver may return several hypotheses and leave the choice to ChampionScorer, so "geometry proposes, features dispose" is reachable as pure composition, with no new scoring code.

`feature_aware_score` is the *mean* cosine over inliers, not paper Eq. 5 (fixed `|P_T|` denominator). The count term arrives separately as `s_icp`. GPU RANSAC's `fitness="feature"` is the Eq. 5 form; that is a different function.

- `solvers.Open3DFeatureRansacSolver` — Open3D's C++ correspondence RANSAC. `n_restarts>1` emits several geometrically-ranked hypotheses; the feature-aware scorer re-ranks the survivors. Open3D draws from a global RNG it does not seed: pass `seed=...` (or `bop_eval --seed`) for a reproducible run; `seed=None` is the unseeded default. That seed stays *out* of the encoder cache key — it moves poses, not features.
- `solvers.GPURansacSolver` — batched RANSAC (vectorised triplet sampling + batched Kabsch/SVD; CPU or CUDA) with a selectable `fitness`, so feature agreement can sit **inside** hypothesis selection, not only after it. `"geometric"` ranks by inlier count. `"feature"` uses the paper's Eq.5 `Σ_inlier cos(f_q,f_t) / |P_T|`: the denominator is the **fixed** sparse-target count, never the inlier count. Features are the **w=1** canonical space.
- `solvers.TeaserSolver` — TEASER++ (Yang, Shi & Carlone, T-RO 2021). Prunes the correspondence pool with a pairwise TIM max-clique and solves rotation by GNC-TLS; deterministic, no RNG. Correspondences come from the same Eq.3 per-target top-k cosine NN pool as `GPURansacSolver` (w=1 features); `tau_inlier` doubles as TEASER's noise bound. The import is deferred to `.solve`, so construction is dep-light.

The comparison that ranks the first three against each other is [examples/solver_swap_demo.py](examples/solver_swap_demo.py). Default solver stays `o3d`; the others are independent configurations. That ranking is not a performance claim for popoe, and a raw rotation-angle median on a near-50/50 flip distribution is not a number to cite.

## Segmentation backends

Every entry satisfies the same `Segmentor` protocol and stamps its origin into `Detection.source`. **File** backends replay an artefact another process wrote; **live** backends run the models themselves. How-to for each source lives in [CNOS.md](CNOS.md), [MUSE.md](MUSE.md), [NIDS_NET.md](NIDS_NET.md), [SAM6D.md](SAM6D.md).

| Implementation | `source` | Kind |
|----------------|----------|------|
| `segmentor_detections.BOPDetectionsSegmentor` | `bop-detections`, or per-source in a union | file — **evaluated**; one JSON or a named-source union |
| `BOPDetectionsSegmentor(..., source="cnos")` | `cnos` | file — official CNOS / CNOS-FastSAM producer, or public BOP default detections |
| `BOPDetectionsSegmentor(..., source="sam6d")` | `sam6d` | file — SAM-6D ISM artefacts |
| `BOPDetectionsSegmentor(..., source="nids")` | `nids` | file — NIDS-Net artefacts |
| `segmentor_muse.MuseDetectionsSegmentor` | `muse-repro` | file — replay of dumped MUSE masks |
| `segmentor_muse.MuseSegmentor` | `muse-repro` | live — GroundingDINO→SAM2→DINOv2; also its own producer |
| `segmentor_cnos_lab.CNOSLabSegmentor` | `cnos-lab` | live — depth-size-gated foreground-patch CNOS |
| `segmentor.SAMSegmentor` | `sam2-amg` | live — SAM2.1 automatic mask generator, class-agnostic |
| `segmentor.DepthSegmentor` | `depth-cc` | live — depth connected components; no model, no GPU |

CNOS-FastSAM, SAM-6D ISM and NIDS-Net all publish the same artefact — a detections JSON — so they are not separate pose-backend code paths, only different named producers. Official checkouts are pinned under `external/` for source provenance but still run in separate environments; popoe-side adapters consume the files. `segmentor_detections.DetectionSource` `(name, path)` is the config handle: select a backend by name and compose several into one `BOPDetectionsSegmentor`.

`topk` is per `(source, label)`, so a top-M union keeps M candidates **per source**. The union across sources is unfiltered: `iou_dedupe` is scoped per source, two backends proposing the same region both survive, and the feature-aware scorer disposes. Official source names (`cnos`, `muse`) are reserved for official artefacts; reimplementation outputs use `cnos-lab` / `muse-repro`. The detections loader hardens stringified records and both RLE forms; a type error is loud (`"1" in [1]` is the silent miss it prevents).

MUSE occupies both forms at once (`MuseSegmentor` live, `MuseDetectionsSegmentor` file replay) because there is no public producer to adapt. Two design points follow from scoring classes *jointly* rather than independently:

- **Classes are registered up front.** `Segmentor.segment` is a per-object contract, but MUSE's relative score is a softmax across all candidate classes. So the segmentor computes a `(proposal x class)` score matrix once and serves one column per call. A single registered class makes that score the constant 1 and silently reduces the method to `beta * S_abs`, so it must be asked for explicitly (`allow_single_class=True`).
- **Proposals are per-frame.** Grounding DINO + SAM2 would otherwise re-run for every object in one image. Results are memoised by frame content, never by `scene_id`/`im_id` (real captures leave those at -1) — the same content-addressing invariant the cache follows.

SAM-6D's ISM half is a detections producer like the others. Its PEM half is an external **full pose** producer, so it is not a `Segmentor` at all: `segmentor_sam6d.SAM6DPemResultsCoarseEstimator` adapts PEM outputs to `PoseHypothesis` through the separate `CoarseEstimator` contract.

## Verification

- **Adapter fidelity** — `examples/bop_eval.py` is the evaluated composition. `examples/freezev2_monolith.py` is a byte-identity check on identical arrays (fixed RANSAC seed + deterministic ICP).
- **Fusion byte-identity & Protocol wiring** — [tests/](tests/), GPU-free (numpy + scikit-learn), run with `pytest`.

## Eval invariants

`examples/bop_eval.py` owns the BOP loop (one query encode per object, w=1 extraction, `--cand-csv`, confusable-object arbitration). A finished target emits exactly `inst_count` rows (champions plus zero-row padding). Resume classifies by row count alone. Partial targets' stale rows are dropped by atomic CSV rewrite before re-run. Local AR/VSD helpers score one-row-per-target and hard-fail on multi-instance CSVs; proper 1–1 assignment is bop_toolkit.

Bare `except` in the eval loop is forbidden: real bugs must not become zero rows indistinguishable from "object not found". Renderer "depth" is camera-space z, never `1/(triangle_id)`. `--grid` and `POPOE_TARGET_GRID` must agree; the effective value is what the cache key records. Template banks and the Pipeline query cache key by `(obj_id, mesh_path)` — BOP ids are unique per dataset, not globally.

Tests that touch fusion knobs must not inherit a dirty shell: `POPOE_VIS_DIM` in the environment changes fused width.

## Visual PCA and extraction weight

PCA component signs are arbitrary per fit. Re-encoding a query without the cached basis scrambles cosine similarity against cached targets. Guards that must stay:

- component-sign canonicalisation after fit
- query features + fitted PCA cached together
- deterministic query sampling (`seed=obj_id`)
- `install_pca(None)` raises — `fuse()` would otherwise fit a target basis and compare across two unrelated spaces
- `Pipeline.run` reinstalls `meta["pca_vis"]` before every target encode when the encoder exposes `install_pca`

A query that needed no reduction hands over `pca_vis="identity"`, not `None`. Arrays without a PCA sidecar are not a cache hit; the sidecar is written first so the `.npz` is the commit marker.

`scale_vis` and `ChampionScorer.s_feat_1` are specified against w=1 features. Extraction must pin `fusion.vis_weight = 1.0`; an env default of 0.5 leaking in makes every sweep weight half its label. `scale_vis` splits fused `[vis | geo]` at `vis_dim` from the caller that knows the features (`enc_cfg['vis_dim']` / `pca_vis.n_components`), not from `POPOE_VIS_DIM`. That env var is only the fusion default. `vis_dim=None` keeps equal halves, which is the geo-matched mainline (64-D + 64-D). A wrong boundary is refused, not guessed.

## Stage caching (config-addressed)

Because stages are separable, their outputs are cacheable — `popoe.cache` keys every stage output by a fingerprint of (stage config, input content, and the keys of any upstream fits it depends on). Same configuration → automatic reuse; changing a knob invalidates exactly the entries it should.

Three invariants:

1. **Fitted state is part of the key.** The target-feature key includes the query key, because the query's fitted visual PCA defines the basis the target features live in.
2. **Content addressing, not positional indices.** A mask's identity is a hash of its pixels, never its index in a detection list.
3. **Every feature-changing knob is config.** The key records the effective target grid, DINO layer, crop/fill/canon settings, render backend, geometric backbone, dGeDi mode, GeDi path, and, when FPFH is active, the FPFH radii, voxel, normal and orientation settings. Missing one turns a sweep into a stale-feature replay.

Changing a keyed knob invalidates existing caches; that is intended. Solver `seed` is not an encoder knob (see [§Pluggability](#pluggability--the-posesolver-stage)).
