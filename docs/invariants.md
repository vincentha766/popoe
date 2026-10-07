# Invariants

Properties that must hold, and the failure that follows when one is removed. These concern the behaviour of a running pipeline rather than its seams; the seams are in [ARCHITECTURE.md](../ARCHITECTURE.md).

Each entry states a rule together with the failure it prevents. Most of them guard against a silent wrong answer rather than a crash, which is why the consequence is documented alongside the rule.

## No silent backend substitution

A stage whose backend is missing raises `interfaces.BackendUnavailable` (`SegmentorUnavailable`, `RendererUnavailable`). It never substitutes a weaker method under the same name.

- A runtime failure, such as CUDA OOM or a corrupt mesh, propagates rather than degrading the result.
- Substitution is the caller's policy: compose an explicit chain, then read `Detection.source` to determine which implementation ran.
- Any parameter that selects a method, such as `render_backend` or a segmentor's `source`, is part of the stage configuration and belongs in the cache key.

Two different methods behind one name make results unattributable, since the logs continue to report the method that was requested. They also corrupt the config-addressed cache, because the key fingerprints the configuration rather than the implementation that executed.

## Cache keys fingerprint config and content

`popoe.cache` keys every stage output by a fingerprint of the stage configuration, the input content, and the keys of any upstream fits it depends on. An unchanged configuration is reused automatically, and a changed parameter invalidates exactly the affected entries. Three conditions must hold:

1. **Fitted state is part of the key.** The target-feature key includes the query key, because the query's fitted visual PCA defines the basis the target features live in.
2. **Content addressing, not positional indices.** A mask's identity is a hash of its pixels, never its index in a detection list.
3. **Every parameter that changes features is part of the configuration.**
   The key records the effective target grid, DINO layer, crop/fill/canon settings, render backend, geometric backbone, dGeDi mode, GeDi path, and — when FPFH is active — the FPFH radii, voxel, normal and orientation settings.

If one such parameter is omitted, a sweep becomes a replay over stale features. Changing a keyed parameter invalidates existing caches, which is the intended behaviour.

The solver `seed` is deliberately excluded from the encoder cache key, because it affects poses rather than features.

## Visual PCA is per object and must be reinstalled

PCA component signs are arbitrary for each fit, so re-encoding a query without the cached basis invalidates cosine similarity against cached targets.
Accuracy on texture-dependent objects collapses while geometry-dominated objects are largely unaffected, which is what makes the failure easy to overlook.

- component-sign canonicalisation after fit;
- query features and fitted PCA cached together;
- deterministic query sampling (`seed=obj_id`);
- `install_pca(None)` raises, since otherwise `fuse()` would fit a target basis and compare features across two unrelated spaces;
- `Pipeline.run` reinstalls `meta["pca_vis"]` before every target encode when the encoder exposes `install_pca`.

A query that required no reduction reports `pca_vis="identity"` rather than `None`. Arrays without a PCA sidecar do not count as a cache hit, and the sidecar is written first so that the `.npz` serves as the commit marker.

## Extraction is pinned at visual weight 1

`scale_vis` and `ChampionScorer.s_feat_1` are specified against w=1 features, so extraction must fix `fusion.vis_weight = 1.0`. If an environment default of 0.5 is applied instead, every sweep runs at half its nominal weight.

`scale_vis` splits fused `[vis | geo]` at `vis_dim`, taken from the caller that knows the features (`enc_cfg['vis_dim']` or `pca_vis.n_components`) rather than from `POPOE_VIS_DIM` — that env var is only the fusion default.
`vis_dim=None` keeps equal halves, the geometry-matched configuration of 64-D + 64-D. An inconsistent boundary is rejected rather than inferred.

Tests that exercise fusion parameters must not inherit environment state from the shell, since `POPOE_VIS_DIM` changes the fused feature width.

## Eval loop

`examples/bop_eval.py` implements the BOP loop: one query encode per object, w=1 extraction, `--cand-csv` dumps, and arbitration between confusable objects.

- A completed target emits exactly `inst_count` rows: the selected hypotheses plus zero-row padding. Resume classifies targets by row count alone, and a partial target's stale rows are removed by an atomic CSV rewrite before it is re-run.
- The local AR/VSD helpers score one row per target and fail on multi-instance CSVs. One-to-one assignment is performed by bop_toolkit.
- A bare `except` in the eval loop is not permitted, because a genuine defect would then produce zero rows that are indistinguishable from "object not found".
- Renderer "depth" is camera-space z, never `1/(triangle_id)`.
- `--grid` and `POPOE_TARGET_GRID` must agree, and the effective value is the one recorded in the cache key.
- Template banks and the `Pipeline` query cache key by `(obj_id, mesh_path)`, because BOP ids are unique within a dataset rather than globally.
