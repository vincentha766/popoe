# Architecture

popoe factors 6-DoF pose into a **method** (`PoseMethod.run(scene, obj)`) and **optional stage** Protocols in [src/popoe/interfaces.py](src/popoe/interfaces.py).
A method uses only the stages its graph needs, and an implementation only needs matching method signatures — there is no base class and no registration step.

This file covers the seams.
The guards that keep a composed run correct are in [docs/invariants.md](docs/invariants.md), and the external producers behind the segmentation stage are in [docs/sources/](docs/sources/README.md).

## Graphs

Library entry: `PoseMethod.run(scene, obj) → PoseHypothesis | None`.

Correspondence graph (`Pipeline` / `CorrespondencePipeline`), built by `make_correspondence_pipeline` in `popoe.freeze.recipes`:

```
ObjectModel (CAD) ─┬─ QueryEncoder ──────────── q, CanonFrame ─┐
                   ├─ Segmentor ─ Detection ─┐                 │
Scene (RGB-D, K) ──┴─────────────────────────┴─ TargetEncoder ─┴─ PoseSolver ─ PoseRefiner* ─ PoseScorer ─ Selector ─ (R, t)
```

Estimator graph (`DirectPoseMethod`), for external full-pose producers such as SAM-6D PEM:

```
Scene, ObjectModel ─ (Segmentor?) ─ CoarseEstimator ─ GeometricRefiner* ─ Selector ─ (R, t)
```

After encode, both `Pipeline.run` and `examples/bop_eval.py` call `correspond_pair` (solve → refine* → score).
The runner still owns the disk cache, the visual-weight sweep, multi-instance NMS, and resume.

`examples/bop_eval.py` is the evaluated BOP loop.
It builds a per-object `Pipeline` with `make_correspondence_pipeline`, then scores each encoded pair with `correspond_pair`.
Its default flags are the tuned Open3D identity rather than a paper-faithful freeze — see [README.md](README.md#three-identities).

## Stages

| Stage | Protocol | Reference implementation |
|-------|----------|--------------------------|
| Segmentation | `Segmentor` | `segmentor_detections.BOPDetectionsSegmentor` (evaluated) — more in [§Segmentation backends](#segmentation-backends) |
| Query features | `QueryEncoder` | `freeze.adapters.FreeZeQueryEncoder` (DINOv2 visual + `PointDescriptor` geometric branch) |
| Target features | `TargetEncoder` | `freeze.adapters.FreeZeTargetEncoder` |
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

## Cross-cutting data (conventions live in one place)

`Scene`, `ObjectModel`, and `CanonFrame` are built once and threaded through the pipeline.
They carry the conventions that would otherwise be re-derived per module and drift apart.
`FrameManifest` sits one step earlier, as the file-level input boundary.

**Units.** Mesh vertices are in mm. Depth-unprojected points and the output `t` are in metres, and BOP CSVs convert back to mm at the edge.

**Frame I/O.** `FrameManifest` is the file boundary for RGB, depth, `K`, and an optional detections JSON.
Detections stay 2D (masks and scores). Depth stays with the frame loader and is converted to metres before `Scene` is constructed.

**Canonicalisation.** `CanonFrame` encodes `pts_canon = (pts - center) * scale`, with `center = 0` and `scale = 1 / max_extent` of the query sampled cloud rather than the BOP diameter.
GeDi was trained at roughly 1 m, so the object is rescaled to roughly 1 m.
The frame is an output of query encoding, since it depends on the sampled points, and it is reused on the target side.

## Design rationale (why these seams)

### Fusion is its own component

`[w·L2(PCA(f_vis)), L2(f_geo)]` used to be copy-pasted inside both encoders.
Extracting `DinoGeDiFusion` turns the whole pure-geometric / pure-visual / fused ablation into a one-liner, `DinoGeDiFusion(vis_weight=0.0 | 1.0 | ...)`.

It also lets query and target share one fusion instance, so the visual PCA fit on the query side is reused on the target side.

### The geometric branch is a component too

The FreeZe recipe uses GeDi, but the encoders call only `PointDescriptor.compute(pts, pcd)`.
That makes FPFH a proper hand-crafted control and dGeDi a fast learned control, without touching query/target encoding, fusion, solving, or scoring.

Descriptor radii are in canonical units, where object extent is about 1.0, so GeDi's `r_lrf` and FPFH's radii are directly comparable.
Role-aware descriptors go through `descriptors.describe(..., role="query"|"target")`; role-blind ones keep the two-argument form.

### Scoring is a stage, not part of the refiner

`PoseScorer` owns the whole feature-scoring concern: the fine re-score at the refined pose, and how the evidence combines.

The combination rule belongs to the implementation rather than the pipeline.
`FreeZeScorer` reproduces the paper's `s_coarse·s_fine·s_icp`. The evaluated `ChampionScorer` uses `s_icp · s_feat_1 · metric_fit`, with `s_coarse` as an opt-in per-dataset factor.

`ICPRefiner` only moves geometry and reports `s_icp`, so a new solver or refiner never re-implements the scoring rule.
The RANSAC-internal inlier score stays inside the solver, where it ranks hypotheses rather than producing a final score.

### A solver only proposes; the scorer disposes

See [§Pluggability](#pluggability--the-posesolver-stage).

### No stage hides a fallback

A stage whose backend is missing raises `interfaces.BackendUnavailable`.
Substitution is the caller's policy, and whatever ran is recorded in `Detection.source`.
The full rule and its consequences: [docs/invariants.md](docs/invariants.md#no-hidden-fallbacks).

### Separable stages are cacheable stages

`popoe.cache` keys every stage output by a fingerprint of the stage config, the input content, and the keys of any upstream fits it depends on.
The same configuration then reuses work, and a changed knob invalidates exactly the entries it should.
The three parts that have to hold: [docs/invariants.md](docs/invariants.md#cache-keys-fingerprint-config-and-content).

## Pluggability — the PoseSolver stage

Three `PoseSolver` implementations run through the identical encoders → refiner → scorer chain.
A solver may return several hypotheses and leave the choice to `ChampionScorer`, so "geometry proposes, features dispose" is reachable as pure composition, with no new scoring code.

One naming caution before the table of solvers. `feature_aware_score` is the *mean* cosine over inliers, which is not paper Eq. 5 with its fixed `|P_T|` denominator; the count term arrives separately as `s_icp`.
GPU RANSAC's `fitness="feature"` is the Eq. 5 form, and that is a different function.

**`solvers.Open3DFeatureRansacSolver`** — Open3D's C++ correspondence RANSAC.
With `n_restarts>1` it emits several geometrically-ranked hypotheses, and the feature-aware scorer re-ranks the survivors.
Open3D draws from a global RNG it does not seed, so pass `seed=...` (or `bop_eval --seed`) for a reproducible run; `seed=None` is the unseeded default.
That seed stays outside the encoder cache key, because it moves poses rather than features.

**`solvers.GPURansacSolver`** — batched RANSAC: vectorised triplet sampling plus batched Kabsch/SVD, on CPU or CUDA.
Its selectable `fitness` lets feature agreement sit inside hypothesis selection instead of only after it.
`"geometric"` ranks by inlier count. `"feature"` uses the paper's Eq. 5 `Σ_inlier cos(f_q,f_t) / |P_T|`, where the denominator is the fixed sparse-target count rather than the inlier count.
Features are the w=1 canonical space.

**`solvers.TeaserSolver`** — TEASER++ (Yang, Shi & Carlone, T-RO 2021).
It prunes the correspondence pool with a pairwise TIM max-clique and solves rotation by GNC-TLS, deterministically and with no RNG.
Correspondences come from the same Eq. 3 per-target top-k cosine NN pool as `GPURansacSolver` (w=1 features), and `tau_inlier` doubles as TEASER's noise bound.
The import is deferred to `.solve`, so construction stays dep-light.

[examples/solver_swap_demo.py](examples/solver_swap_demo.py) is the comparison that ranks the three against each other.
The default solver stays `o3d`; the others are independent configurations.

That ranking is not a performance claim for popoe. A raw rotation-angle median on a near-50/50 flip distribution is also not a number worth citing.

## Segmentation backends

Every entry satisfies the same `Segmentor` protocol and stamps its origin into `Detection.source`.
A **file** backend replays an artefact another process wrote; a **live** backend runs the models itself.
Per-source how-to, environments and artefact provenance: [docs/sources/](docs/sources/README.md).

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

CNOS-FastSAM, SAM-6D ISM and NIDS-Net all publish the same artefact, a detections JSON, so they are different named producers rather than separate pose-backend code paths.
Official checkouts are pinned under `external/` for source provenance but still run in their own environments, with popoe-side adapters consuming the files.
`segmentor_detections.DetectionSource` `(name, path)` is the config handle: select a backend by name, and compose several into one `BOPDetectionsSegmentor`.

`topk` is per `(source, label)`, so a top-M union keeps M candidates per source.
The union across sources is unfiltered: `iou_dedupe` is scoped per source, two backends proposing the same region both survive, and the feature-aware scorer decides between them.
The detections loader hardens stringified records and both RLE forms, and raises loudly on a type error — `"1" in [1]` is the silent miss it prevents.

MUSE occupies both forms at once, `MuseSegmentor` live and `MuseDetectionsSegmentor` for file replay, because there is no public producer to adapt.
Two design points follow from scoring classes *jointly* rather than independently.

**Classes are registered up front.** `Segmentor.segment` is a per-object contract, but MUSE's relative score is a softmax across all candidate classes, so the segmentor computes a `(proposal x class)` score matrix once and serves one column per call.
A single registered class makes that score the constant 1, which reduces the method to `beta * S_abs`, so that configuration has to be asked for explicitly with `allow_single_class=True`.

**Proposals are per-frame.** Grounding DINO + SAM2 would otherwise re-run for every object in one image.
Results are memoised by frame content rather than by `scene_id`/`im_id`, which real captures leave at -1 — the same content-addressing invariant the cache follows.

SAM-6D's ISM half is a detections producer like the others.
Its PEM half produces full poses, so it is not a `Segmentor` at all: `segmentor_sam6d.SAM6DPemResultsCoarseEstimator` adapts PEM outputs to `PoseHypothesis` through the separate `CoarseEstimator` contract.

## Verification

**Evaluated composition** — `examples/bop_eval.py` (ChampionScorer + Open3D). Paper-side flags are opt-in; see [README.md](README.md#three-identities).

**Adapter parity oracle** — `examples/pipeline_selfcheck.py` checks `RansacSolver` + `FreeZeScorer` against `examples/freezev2_monolith.py` on identical arrays, with a fixed RANSAC seed and deterministic ICP. It is not the BOP eval loop.

**Fusion byte-identity and Protocol wiring** — [tests/](tests/), GPU-free (numpy + scikit-learn), run with `pytest`.

The invariants these checks defend, and the failure each one prevents, are listed in [docs/invariants.md](docs/invariants.md).
