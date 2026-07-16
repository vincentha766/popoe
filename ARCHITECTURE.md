# Architecture

popoe factors a training-free 6-DoF pose pipeline into **swappable stages**, each
a `typing.Protocol` in [src/popoe/interfaces.py](src/popoe/interfaces.py). An
implementation only needs matching method signatures — no base class, no
registration — so stages stay decoupled and any one can be re-implemented alone.

## Stages

```
ObjectModel (CAD) ─┐
                   ├─ Segmentor ─ QueryEncoder ─┐
Scene (RGB-D, K) ──┘            TargetEncoder ──┴─ PoseSolver ─ PoseRefiner* ─ PoseScorer ─ Selector ─ (R, t)
```

| Stage | Protocol | Reference implementation |
|-------|----------|--------------------------|
| Segmentation | `Segmentor` | `segmentor_detections.BOPDetectionsSegmentor` (evaluated; single file or a named-source union — see below); `segmentor_cnos.CNOSSegmentor` / `.DinoWindowSegmentor`; `segmentor.SAMSegmentor` / `.DepthSegmentor`; `adapters.PrecomputedSegmentor` |
| Query features | `QueryEncoder` | `adapters.FreeZeQueryEncoder` (DINOv2 + GeDi) |
| Target features | `TargetEncoder` | `adapters.FreeZeTargetEncoder` |
| Fusion | `FeatureFusion` | `fusion.DinoGeDiFusion` |
| Pose solve | `PoseSolver` | `adapters.RansacSolver`; `solvers.Open3DFeatureRansacSolver` |
| Refine | `PoseRefiner` | `adapters.ICPRefiner` |
| Score | `PoseScorer` | `adapters.FreeZeScorer` |
| Select | `Selector` | `adapters.BestScoreSelector` |
| Metrics | `Metric` | `metrics.vsd`, `metrics.ar` |

The reference control flow is `interfaces.Pipeline.run`.

## Cross-cutting data (conventions live in one place)

`Scene`, `ObjectModel`, and `CanonFrame` are built once and threaded through
every stage, carrying the conventions that would otherwise be re-derived per
module and drift:

- **Units** — mesh vertices in mm; depth-unprojected points and output `t` in
  **metres** (BOP CSVs convert back to mm at the edge).
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
- **Scoring is a stage, not baked into the refiner.** `PoseScorer` owns the whole
  feature-scoring concern (fine re-score + the `s_coarse·s_fine·s_icp`
  combination). `ICPRefiner` only moves geometry and reports `s_icp`. So a new
  solver or refiner never re-implements the scoring rule. (Note: the RANSAC-internal
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

Two things that costs:

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

## Pluggability proven — a second PoseSolver

Two independent `PoseSolver` implementations run through the identical
encoders→refiner→scorer→selector chain, changing one line:

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
proposes, features dispose." A robust backend (TEASER++, MAC) would slot in the
same way.

## File-based detection backends (CNOS / SAM-6D / NIDS)

CNOS-FastSAM, SAM-6D ISM and NIDS-Net all publish the same artefact — a
BOP-format detections JSON — so they are not separate code paths, only
different files under different names. `segmentor_detections.DetectionSource`
`(name, path)` is the config handle: select a backend BY NAME and compose
several into one `BOPDetectionsSegmentor` to reproduce FreeZe-style multi-source
segmentation.

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

## Verification

- **Adapter fidelity** — [examples/pipeline_selfcheck.py](examples/pipeline_selfcheck.py):
  the adapter chain reproduces the inline `FreeZeV2.estimate_pose` body to ~1e-15
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

Two invariants, both learned from real incidents (see ISSUES.md):

1. **Fitted state is part of the key.** The target-feature key includes the
   QUERY key, because the query's fitted visual PCA defines the basis the
   target features live in. (Violation: silent cross-run basis mismatch;
   texture-reliant objects crater.)
2. **Content addressing, not positional indices.** A mask's identity is a
   hash of its pixels, never its index in a detection list. (Violation:
   pooling reorders the list and a *different* mask's features load.)
