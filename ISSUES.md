# Known issues — rules that constrain the code

The rule each incident taught. Seams belong in
[ARCHITECTURE.md](ARCHITECTURE.md).

## No hidden fallbacks

A stage whose backend is missing raises `BackendUnavailable`. It never
substitutes a weaker method under the same name. Substitution is the
caller's policy, recorded in `Detection.source`. Runtime failures
(CUDA OOM, corrupt mesh) propagate.

The original case: a CNOS path silently chained SAM2 → sliding window →
depth blobs, then merge-sorted depth-blob *area fractions* among DINO
cosines. The same shape later showed up in fusion (`install_pca(None)`
re-fitted on target data; `fuse()` truncated instead of projecting).
A platform rule is only real where someone enforced it.

## Cache keys fingerprint config and content

1. Fitted state is part of the key. Cached target features are only
   valid with the query visual-PCA that produced their basis, so the
   target key includes the query key.
2. A mask's identity is a hash of its pixels, never its index in a
   detection list.
3. Every feature-changing knob is config: grid, DINO layer, crop / fill
   / canon, render backend, geometric backbone, GeDi path, FPFH radii.

Missing a knob turns a sweep into a stale-feature replay. Changing a
keyed knob invalidates existing caches; that is intended.

The seed on `Open3DFeatureRansacSolver` stays *out* of the encoder
cache key: it moves poses, not features.

## Visual PCA is per object and must be reinstalled

PCA component signs are arbitrary per fit. Re-encoding a query without
the cached basis scrambles cosine similarity against cached targets
(texture-reliant objects crater; geometry-strong ones survive).

Fixes that must stay:

- component-sign canonicalisation after fit
- query features + fitted PCA cached together
- deterministic query sampling (`seed=obj_id`)
- `install_pca(None)` raises — `fuse()` would otherwise fit a target
  basis and compare across two unrelated spaces
- `Pipeline.run` reinstalls `meta["pca_vis"]` before every target
  encode when the encoder exposes `install_pca`

A query that needed no reduction hands over `pca_vis="identity"`, not
`None`. Arrays without a PCA sidecar are not a cache hit; the sidecar
is written first so the `.npz` is the commit marker.

## Extraction is pinned at visual weight 1

`scale_vis` and `ChampionScorer.s_feat_1` are specified against w=1
features. Extraction must pin `fusion.vis_weight = 1.0`; an env default
of 0.5 leaking in makes every sweep weight half its label.

`scale_vis` splits fused `[vis | geo]` at `vis_dim` from the caller
that knows the features (`enc_cfg['vis_dim']` / `pca_vis.n_components`),
not from `POPOE_VIS_DIM`. That env var is only the fusion default.
`vis_dim=None` keeps equal halves, which is the geo-matched mainline
(64-D + 64-D). A wrong boundary is refused, not guessed.

## Union is per-source; the scorer disposes

Top-M is per `(source, label)`. `iou_dedupe` is scoped per source.
Two backends proposing the same region both survive. Official names
(`cnos`, `muse`, …) are reserved for official artefacts; lab outputs
use `cnos-lab` / `muse-repro`.

The detections loader hardens stringified records and both RLE forms.
A type error is loud; `"1" in [1]` is the silent miss it prevents.

## Resume is a writer invariant

A finished target emits exactly `inst_count` rows (champions plus
zero-row padding). Resume classifies by row count alone. Partial
targets' stale rows are dropped by atomic CSV rewrite before re-run.
Local AR/VSD helpers score one-row-per-target and hard-fail on
multi-instance CSVs; proper 1–1 assignment is bop_toolkit.

Bare `except` in the eval loop is forbidden: real bugs must not become
zero rows indistinguishable from "object not found".

## Geometry proposes; features dispose

A solver may emit several hypotheses; `ChampionScorer` chooses.
`feature_aware_score` is the *mean* cosine over inliers, not paper
Eq. 5 (fixed `|P_T|` denominator). Confusing the two was measured at
−31 pt. The count term arrives separately as `s_icp`. GPU RANSAC's
`fitness="feature"` is the Eq. 5 form; that is a different function.

Open3D RANSAC draws from a global RNG it does not seed. Pass
`Open3DFeatureRansacSolver(seed=...)` (or `bop_eval --seed`) for any
cited number; `seed=None` is the historical unseeded mainline.

Do not cite a raw rotation-angle median on a near-50/50 flip
distribution. `examples/solver_swap_demo.py` compares the shipped
solvers; that ranking is not a performance claim for popoe.

## Eval swallowed failures and under-keyed caches (do not regress)

Renderer "depth" is camera-space z, never `1/(triangle_id)`.
`--grid` and `POPOE_TARGET_GRID` must agree; the effective value is
what the cache key records. Template banks and the Pipeline query
cache key by `(obj_id, mesh_path)` — BOP ids are unique per dataset,
not globally.

Tests that touch fusion knobs must not inherit a dirty shell:
`POPOE_VIS_DIM` in the environment changes fused width.
