# Architecture

popoe factors a 6-DoF pose pipeline into **swappable stages**, each
a `typing.Protocol` in [src/popoe/interfaces.py](src/popoe/interfaces.py). An
implementation only needs matching method signatures — no base class, no
registration — so stages stay decoupled and any one can be re-implemented alone.

## Stages

```
ObjectModel (CAD) ─┬─ QueryEncoder ──────────── q, CanonFrame ─┐
                   ├─ Segmentor ─ Detection ─┐                 │
Scene (RGB-D, K) ──┴─────────────────────────┴─ TargetEncoder ─┴─ PoseSolver ─ PoseRefiner* ─ PoseScorer ─ Selector ─ (R, t)
```

| Stage | Protocol | Reference implementation |
|-------|----------|--------------------------|
| Segmentation | `Segmentor` | `segmentor_detections.BOPDetectionsSegmentor` (evaluated) — 12 more in [§Segmentation backends](#segmentation-backends) |
| Query features | `QueryEncoder` | `freeze.adapters.FreeZeQueryEncoder` (DINOv2 visual + `PointDescriptor` geometric branch) |
| Target features | `TargetEncoder` | `freeze.adapters.FreeZeTargetEncoder` |
| Geometric descriptors | `PointDescriptor` | GeDi default via `freeze.feature_extractor.load_gedi`; `descriptors.FPFHDescriptor`; dGeDi via `POPOE_GEOM_BACKBONE` |
| Fusion | `FeatureFusion` | `freeze.fusion.DinoGeDiFusion` |
| Pose solve | `PoseSolver` | `solvers.Open3DFeatureRansacSolver` (default) — 3 more in [§Pluggability proven](#pluggability-proven--the-posesolver-stage) |
| External coarse pose | `CoarseEstimator` | `segmentor_sam6d.SAM6DPemResultsCoarseEstimator` over already-written PEM results |
| Refine | `PoseRefiner` | `adapters.ICPRefiner` |
| Score | `PoseScorer` | `freeze.adapters.FreeZeScorer`; `scoring.ChampionScorer` (evaluated) |
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
  `s_coarse` an opt-in per-dataset factor (helps YCB-V, hurts LM-O).
  `ICPRefiner` only moves geometry and reports `s_icp`. So a new solver or
  refiner never re-implements the scoring rule. (Note: the RANSAC-internal
  inlier score stays inside the solver — that's hypothesis ranking, not final
  scoring.)
- **A solver only PROPOSES; the scorer DISPOSES.** See below.
- **No stage hides a fallback.** A stage whose backend is missing raises
  `interfaces.BackendUnavailable` — it never quietly substitutes a weaker method.
  See below.

## The availability contract (no hidden fallbacks)

Two different methods behind one name is a bug, not a convenience. It used to be
the norm here: `CNOSSegmentor.segment` caught a SAM2 load failure and silently
ran a sliding-window variant, which silently swapped its own mask generator, and
then topped the list up with depth blobs whose "score" was a mask **area
fraction** mixed in among DINO **cosine similarities** — a blob covering 40% of
the frame outranked a real template match at 0.35. `SAMSegmentor` and
`get_renderer` did the same thing more quietly.

That costs two things:

1. **The result becomes unattributable.** A run on a box without the SAM2
   checkpoint produced depth-blob masks while every log line and config still
   said "CNOS".
2. **It poisons the config-addressed cache** (see below), whose key fingerprints
   the config you *asked* for — not the method that silently ran instead. The
   renderer was the live case: nvdiffrast and the trimesh CPU ray-caster produce
   different CAD views, hence different query features, and `render_backend` was
   absent from the key, so a cache built without a GPU was reused on one with it.

So:

- an implementation raises `BackendUnavailable` (`SegmentorUnavailable`,
  `RendererUnavailable`) when a package / checkpoint / device is missing;
- a **runtime** failure (CUDA OOM, corrupt mesh) propagates — "the fallback
  handled it" is how real bugs get buried;
- substitution is the **caller's** policy: compose
  `segmentor.FirstAvailableSegmentor([...])`, then read `chain.last_used` and
  `Detection.source` to see what ran;
- anything that selects a method (`render_backend`, the segmentor's `source`)
  is part of the stage config and belongs **in the cache key**.

## Pluggability proven — the PoseSolver stage

Four `PoseSolver` implementations run through the identical
encoders→refiner→scorer→selector chain, each selected by changing one line. The
original proof was a pair:

- `adapters.RansacSolver` — hand-rolled feature-aware RANSAC.
- `solvers.Open3DFeatureRansacSolver` — Open3D's C++ correspondence RANSAC, added
  as one new file, zero changes elsewhere.

The A/B (see [examples/solver_swap_demo.py](examples/solver_swap_demo.py)) also
surfaces a real finding and its fix by composition alone. On the near-symmetric
mustard bottle (YCB-V obj 5, 5 instances), median rotation error:

| solver | median rot | median trans |
|--------|-----------|--------------|
| `freeze_ransac` | 23.4° | 17.6 mm |
| `open3d` (1 shot) | 42.5° (flips: 94°, 152°) | 19.5 mm |
| `open3d` (`n_restarts=8`) | **23.9°** | 17.9 mm |

One-shot Open3D ranks by geometric inlier fitness and flips on symmetric geometry
the visual features would disambiguate. Emitting several candidates
(`n_restarts=8`) and letting the EXISTING feature-aware `PoseScorer` + `Selector`
pick the feature-best — **no new scoring code** — recovers parity. "Geometry
proposes, features dispose." A robust backend (TEASER++, MAC) slots in the same
way — TEASER++ since has, two headings below.

### A third solver — feature-aware fitness INSIDE selection (the B layer)

Open3D's C++ RANSAC ranks hypotheses by geometric inlier count and cannot take a
custom fitness, so feature agreement can only re-rank the survivors (that is the
A layer — `PoseScorer`). To put feature agreement INSIDE hypothesis selection —
changing which hypotheses survive — `solvers.GPURansacSolver` ports gedi's
batched RANSAC (vectorised triplet sampling + batched Kabsch/SVD; runs on CPU or
CUDA) with a selectable `fitness`:

- `"geometric"` (default) — rank by inlier count; a faithful port whose
  behaviour is verifiable against Open3D alone.
- `"feature"` — the paper's Eq.5 score `Σ_inlier cos(f_q,f_t) / |P_T|`. The
  denominator is the **fixed** sparse-target count `|P_T|`, never the inlier
  count: normalise-by-inlier (mean cosine) lets a few high-similarity spurious
  correspondences beat many true ones (a −31 pt regression in the study). The
  features are the **w=1** canonical space (the same lesson the A-layer S_coarse
  learned).

### A fourth solver — TEASER++ certifiable registration

`solvers.TeaserSolver` wraps TEASER++ (Yang, Shi & Carlone, T-RO 2021) — the
robust backend the Open3D A/B above anticipated. Instead of sampling
minimal triplets, it prunes the correspondence pool with a pairwise
translation-invariant-measurement max-clique and solves rotation by GNC-TLS:
robust to >90% outlier correspondences, deterministic (no RNG, no seed), with
optimality certificates. Correspondences come from the same Eq.3 per-target
top-k cosine NN pool as `GPURansacSolver` (w=1 features); `tau_inlier` doubles
as TEASER's noise bound. Needs `teaserpp_python`, built from source
([MIT-SPARK/TEASER-plusplus](https://github.com/MIT-SPARK/TEASER-plusplus) —
no PyPI wheel); the import is deferred to `.solve`, so construction is
dep-light.

Select with `freeze.recipes.stages_for_object(solver=...)` or `bop_eval --solver
o3d|gpu|gpu-feat|teaser`. The default stays `o3d`, so the evaluated mainline is
unperturbed; the non-default solvers are reported as independent configurations.

## Segmentation backends

Every entry satisfies the same `Segmentor` protocol and stamps its origin into
`Detection.source`. **File** backends replay an artefact another process wrote;
**live** backends run the models themselves.

| Implementation | `source` | Kind |
|----------------|----------|------|
| `segmentor_detections.BOPDetectionsSegmentor` | `bop-detections`, or per-source in a union | file — **evaluated**; one JSON or a named-source union |
| `segmentor_cnos_official.CNOSDetectionsSegmentor` | `cnos` | file — official CNOS / CNOS-FastSAM producer, or public BOP default detections |
| `segmentor_sam6d.SAM6DIsmDetectionsSegmentor` | `sam6d` | file — SAM-6D ISM artefacts |
| `segmentor_nids.NIDSNetDetectionsSegmentor` | `nids` | file — NIDS-Net artefacts |
| `segmentor_muse.MuseDetectionsSegmentor` | `muse-repro` | file — replay of dumped MUSE masks |
| `segmentor_muse.MuseSegmentor` | `muse-repro` | live — GroundingDINO→SAM2→DINOv2; also its own producer |
| `segmentor_cnos_v3.CNOSv3Segmentor` | `cnos-v3` | live — depth-size-gated foreground-patch CNOS |
| `segmentor_cnos.CNOSSegmentor` | `cnos-live` | live — SAM2 AMG proposes, DINOv2 matches |
| `segmentor_cnos.DinoWindowSegmentor` | `dino-window` | live — DINOv2 multi-scale sliding-window matching |
| `segmentor.SAMSegmentor` | `sam2-amg` | live — SAM2.1 automatic mask generator, class-agnostic |
| `segmentor.DepthSegmentor` | `depth-cc` | live — depth connected components; no model, no GPU |
| `segmentor.FirstAvailableSegmentor` | delegates | explicit fallback chain — see the availability contract |
| `adapters.PrecomputedSegmentor` | as supplied | inject fixed masks (tests, ablations) |

### File backends (CNOS / SAM-6D / NIDS)

CNOS-FastSAM, SAM-6D ISM and NIDS-Net all publish the same artefact — a
detections JSON — so they are not separate pose-backend code paths, only
different named producers. Official CNOS, NIDS-Net and SAM-6D are pinned under
`external/` for source provenance but still run in separate
environments/services; `segmentor_cnos_official.CNOSDetectionsSegmentor`,
`segmentor_nids.NIDSNetDetectionsSegmentor`,
`segmentor_nids.adapt_nidsnet_json`, and
`segmentor_sam6d.SAM6DIsmDetectionsSegmentor` are the popoe-side adapters.
Underneath,
`segmentor_detections.DetectionSource` `(name, path)` is the config handle:
select a backend BY NAME and compose several into one `BOPDetectionsSegmentor`
to reproduce FreeZe-style multi-source segmentation.

```python
from popoe.segmentor_detections import BOPDetectionsSegmentor

seg = BOPDetectionsSegmentor(sources={           # or [("nids", p), ...] / "name=path"
    "cnos":  "…/cnos_ycbv.json",
    "sam6d": "…/sam6d_ycbv.json",
    "nids":  "…/nids_wa_sappe_ycbv.json",
}, topk=2)
dets = seg.segment(scene, obj)
dets[0].source        # -> 'cnos' | 'sam6d' | 'nids' — which backend produced it
```

The three CNOS source names (`cnos`, `cnos-v3`, `cnos-live` — see the inventory
above) are **reserved**: only the official producer's artefacts may be filed
under `cnos`, for the same reason `muse` is reserved below.

`topk` is applied per `(source, label)` bucket, so a top-M union keeps M
candidates **per source** (no source crowds out another before scoring), and
every mask carries its origin in `Detection.source` — the same provenance
discipline as the fallback chain. The union across sources is **unfiltered**
(FreeZe's "top-M union without filtering"): `iou_dedupe` is scoped per source,
so two backends proposing the same region both survive and the feature-aware
scorer disposes with every source's evidence intact — a single backend still
drops its own near-duplicates. The single-file form
`BOPDetectionsSegmentor(path)` is unchanged (its masks keep the historical
`bop-detections` tag). The loader (`load_bop_detections`) coerces the
fully-stringified NIDS WA_Sappe variant and decodes both compressed and
uncompressed RLE — see the module docstring.

### MUSE — the backend that has to be both (`segmentor_muse`)

MUSE is the fourth mask source in FreeZeV2's ensemble and the only one with **no
public code and no downloadable masks**, so there is no producer to adapt. What
popoe carries is a reimplementation from the paper (arXiv 2510.17866), and it
occupies both forms at once: `MuseSegmentor` computes masks from pixels like
`CNOSv3Segmentor`, while `muse_records` / `write_muse_detections` dump those same
masks to a detections JSON, after which `MuseDetectionsSegmentor` replays them as
an ordinary named source — GPU-free, unionable, archivable. See [MUSE.md](MUSE.md).

| Source | Meaning |
|--------|---------|
| `muse` | RESERVED for official MUSE artefacts; nothing here writes it |
| `muse-repro` | this reimplementation |

The naming rule is the CNOS rule with more at stake: the study cites MUSE as
evidence that an ensemble member is externally unreproducible, so our own
reimplementation must never wear the official name.

Two design points follow from MUSE scoring classes *jointly* rather than
independently, and neither is optional:

- **Classes are registered up front.** `Segmentor.segment` is a per-object
  contract, but MUSE's relative score is a softmax across all candidate classes.
  So the segmentor computes a `(proposal x class)` score matrix once and serves
  one column per call. A single registered class makes that score the constant 1
  and silently reduces the method to `beta * S_abs` — the availability
  contract's "two methods behind one name" in miniature, so it must be asked for
  explicitly (`allow_single_class=True`).
- **Proposals are per-frame.** Grounding DINO + SAM2 would otherwise re-run for
  every object in one image. Results are memoised by frame CONTENT, never by
  `scene_id`/`im_id` (real captures leave those at -1) — the same content-addressing
  invariant the cache follows.

### SAM-6D PEM is not one of these

SAM-6D's ISM half is a detections producer like the others above. Its PEM half
is an external **full pose** producer, so it is not a `Segmentor` at all:
`segmentor_sam6d.SAM6DPemResultsCoarseEstimator` adapts PEM's BOP CSV or custom
JSON outputs to `PoseHypothesis` through the separate `CoarseEstimator`
contract, keeping it out of the FreeZe feature-solver contract.

## BOP runner invariants

`examples/bop_eval.py` is a composition runner, not a new stage, but several
of its rules are load-bearing for reproducible benchmark runs:

- **One query encode per object.** Queries and fitted visual PCA are cached up
  front. The target cache key includes the query key because target visual
  features live in the query PCA basis.
- **Feature extraction is pinned at w=1.** The visual-weight sweep reweights
  cached `[vis|geo]` features at selection time, and `ChampionScorer`'s
  `s_feat_1` really is a w=1 re-score.
- **Completed targets emit exactly `inst_count` rows.** Champions are written
  first and zero rows pad any missing instances. Resume is therefore a row-count
  invariant: fewer rows means a partial target and those stale rows are dropped
  before re-run.
- **Candidate dumps are the offline interface.** `--cand-csv` records every
  mask x visual-weight hypothesis with its score breakdown and solver name, so
  selection rules can be replayed without re-running DINO/GeDi/FPFH.
- **Confusable-object arbitration is explicit.** YCB-V clamp label pooling is
  the formal path; `--size-select` and `--dual-assign` are score-affecting lab
  paths and require fresh output files.

## Verification

- **Adapter fidelity** — [examples/pipeline_selfcheck.py](examples/pipeline_selfcheck.py):
  the adapter chain reproduces the inline `FreeZeV2.estimate_pose` body
  (`examples/freezev2_monolith.py`) to ~1e-15
  on identical arrays (fixed RANSAC seed + deterministic ICP).
- **Fusion byte-identity & Protocol wiring** — [tests/](tests/), GPU-free
  (numpy + scikit-learn), run with `pytest`.

## Stage caching (config-addressed)

Because stages are separable, their outputs are cacheable — `popoe.cache`
keys every stage output by a fingerprint of (stage config, input CONTENT,
and the keys of any upstream fits it depends on). Same configuration →
automatic reuse; changing a knob invalidates exactly the entries it should.

Measured payoff (reproduction study): reruns skip GeDi+DINO entirely
(registration-only iterations), selection rules are swappable with zero GPU
via the candidate dump, and whole diagnostic investigations run offline
against cached features.

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
