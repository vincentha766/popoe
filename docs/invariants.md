# Invariants

Guards that must stay, and what breaks when one is removed. These are properties of the running system rather than of its seams — the seams are in [ARCHITECTURE.md](../ARCHITECTURE.md).

Each entry is written as a rule plus the failure it prevents, because most of them exist to stop a *silent* wrong answer rather than a crash.

## No hidden fallbacks

A stage whose backend is missing raises `interfaces.BackendUnavailable` (`SegmentorUnavailable`, `RendererUnavailable`). It never substitutes a weaker method under the same name.

- A runtime failure — CUDA OOM, a corrupt mesh — propagates rather than degrading.
- Substitution is the caller's policy: compose an explicit chain, then read `Detection.source` to see what actually ran.
- Anything that selects a method (`render_backend`, a segmentor's `source`) is stage config and belongs in the cache key.

Two different methods behind one name make results unattributable, because the logs still name the method that was asked for. They also poison the config-addressed cache, because the key fingerprints the config rather than the method that ran.

## Cache keys fingerprint config and content

`popoe.cache` keys every stage output by a fingerprint of the stage config, the input content, and the keys of any upstream fits it depends on. Same configuration, automatic reuse; change a knob and exactly the affected entries are invalidated. Three parts have to hold:

1. **Fitted state is part of the key.** The target-feature key includes the query key, because the query's fitted visual PCA defines the basis the target features live in.
2. **Content addressing, not positional indices.** A mask's identity is a hash of its pixels, never its index in a detection list.
3. **Every feature-changing knob is config.** The key records the effective target grid, DINO layer, crop/fill/canon settings, render backend, geometric backbone, dGeDi mode, GeDi path, and — when FPFH is active — the FPFH radii, voxel, normal and orientation settings.

Miss one knob and a sweep becomes a stale-feature replay. Changing a keyed knob invalidates existing caches, which is the intended behaviour.

The solver `seed` is deliberately outside the encoder cache key: it moves poses, not features.

## Visual PCA is per object and must be reinstalled

PCA component signs are arbitrary per fit, so re-encoding a query without the cached basis scrambles cosine similarity against cached targets. Texture-reliant objects crater; geometry-strong ones survive, which is what makes the failure hard to notice.

- component-sign canonicalisation after fit;
- query features and fitted PCA cached together;
- deterministic query sampling (`seed=obj_id`);
- `install_pca(None)` raises — otherwise `fuse()` fits a target basis and compares across two unrelated spaces;
- `Pipeline.run` reinstalls `meta["pca_vis"]` before every target encode when the encoder exposes `install_pca`.

A query that needed no reduction hands over `pca_vis="identity"`, not `None`. Arrays without a PCA sidecar are not a cache hit; the sidecar is written first, so the `.npz` is the commit marker.

## Extraction is pinned at visual weight 1

`scale_vis` and `ChampionScorer.s_feat_1` are specified against w=1 features, so extraction must pin `fusion.vis_weight = 1.0`. An env default of 0.5 leaking in makes every sweep weight half its label.

`scale_vis` splits fused `[vis | geo]` at `vis_dim`, taken from the caller that knows the features (`enc_cfg['vis_dim']` or `pca_vis.n_components`) rather than from `POPOE_VIS_DIM` — that env var is only the fusion default.
`vis_dim=None` keeps equal halves, the geo-matched mainline of 64-D + 64-D, and a wrong boundary is refused rather than guessed.

Tests that touch fusion knobs must not inherit a dirty shell, since `POPOE_VIS_DIM` in the environment changes fused width.

## Eval loop

`examples/bop_eval.py` owns the BOP loop: one query encode per object, w=1 extraction, `--cand-csv` dumps, and confusable-object arbitration.

- A finished target emits exactly `inst_count` rows — champions plus zero-row padding. Resume classifies by row count alone, and a partial target's stale rows are dropped by atomic CSV rewrite before re-run.
- Local AR/VSD helpers score one row per target and hard-fail on multi-instance CSVs. Proper 1–1 assignment is bop_toolkit's job.
- Bare `except` in the eval loop is forbidden, because a real bug would become zero rows indistinguishable from "object not found".
- Renderer "depth" is camera-space z, never `1/(triangle_id)`.
- `--grid` and `POPOE_TARGET_GRID` must agree; the effective value is whatever the cache key records.
- Template banks and the `Pipeline` query cache key by `(obj_id, mesh_path)`, because BOP ids are unique per dataset rather than globally.
