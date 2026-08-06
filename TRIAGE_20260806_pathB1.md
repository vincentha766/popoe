# TRIAGE 2026-08-06 — path B1: silent-wrong-answer hunt before the 18-run freeze

Adversarial read of the formal-run path (`examples/bop_eval.py` + the stages the
four frozen REPRODUCTION.md recipes wire). Hunted the render_rerank
`breakdown.get("tau_icp", 0.03)` class: unwritten keys masked by defaults,
unit/space confusion, guards that pass ambiguous values, asymmetric measurement
between compared candidates, stale cached state, RNG leaks.

Recipes checked against: faithful-cnos / faithful-3way (env pins incl.
`POPOE_CANON_BASIS=diameter`, `POPOE_QUERY_MIN_VIEWS=18`, `--grid 16 --seed 42
--weights 1.0 --use-s-coarse --corr-topk 10 --tau-diameter --icp-dense
--icp-dense-max 3000 --render-rerank`) and tuned-cnos / tuned-4way (all env
unset, `--grid 32`, weight sweep, `--render-rerank`; YCB-V adds `--merge ycbv
--use-s-coarse`).

## Findings

| # | Location | Defect | Concrete wrong-answer scenario | Severity |
|---|---|---|---|---|
| F1 | `examples/bop_eval.py:874` + `src/popoe/interfaces.py:189-192` (`CanonFrame.from_points`) + `src/popoe/freeze/adapters.py:129` (`encode_target` consumes `frame.scale`) | **Query cache HIT rebuilds the canonical frame in the wrong basis.** On a hit, `q.meta["canon_frame"] = CanonFrame.from_points(q.pts)` = `1/extent` of the cached (post-min-views-gate) points. Under the faithful pins the query was ENCODED at `_canon_scale = 1/hull-diameter` (`POPOE_CANON_BASIS=diameter`, feature_extractor.py:666-677). The target encoder then scales every new target cloud by `1/extent` for GeDi while the cached query GeDi features live at `1/diameter` — a 2-35% canonical-scale mismatch (the repo's own LM-O figure for diameter vs extent). The cache key (`enc_cfg["canon_basis"]="diameter"`) claims a basis the features no longer have, so the poisoned target entries are **cached under clean keys** and served to every later run. | Any warm-query / cold-target situation in a faithful arm: (a) **resume after a mid-run crash** — the standard relaunch re-loads all queries from cache, then encodes every remaining image's targets at the wrong scale; half the CSV is scored with mismatched geometric features, no error anywhere; (b) smoke-then-full or "cache may point at a verified warm directory" (REPRODUCTION.md's own instruction) with any target not yet covered; (c) reusing the blessed 10GB void-batch caches with any new mask. Tuned arms are safe only because they leave basis=extent and the gate off, which `from_points` happens to reproduce. | **blocks-pin** (faithful arms). Fix direction: persist the canon scale (or basis) in the query cache entry and refuse/recompute on hit; `from_points` must not be the reconstruction under a non-default basis. Same latent fallback at `src/popoe/interfaces.py:383`. |
| F2 | `src/popoe/scoring.py:131-143` (s_coarse at `R_coarse/t_coarse`) with `src/popoe/freeze/recipes.py:333-338` (chain: ICP then reranker) and `src/popoe/render_rerank.py:284-335` | **`--use-s-coarse` multiplies in a factor measured at a pose the reranker discarded.** `R_coarse/t_coarse` are the pre-ICP pose of the pre-flip lineage; the reranker flips `R` but the breakdown's coarse pose is untouched, so `score = s_icp*s_feat_1*s_coarse` scores the corrected orientation with the coarse-pose cosine of the WRONG orientation. Asymmetric between compared candidates: a rerank-corrected hypothesis carries the low s_coarse it was rescued from, an uncorrected competitor carries a self-consistent one. | Faithful YCB-V/LM-O (and tuned YCB-V): mask A's coarse pose is 180-flipped (s_coarse 0.1), rerank+re-ICP fix it into the best pose in the pool; mask B's mediocre unflipped pose has s_coarse 0.4 and wins the champion slot. The rerank lift and the replay-measured "+2.5 s_coarse on YCB-V" were each measured WITHOUT the other; their product term is unmeasured, and the S1 smoke checker inspects only s_icp symmetry, so this passes every gate. | **must-disclose** (and consider re-measuring s_coarse's sign with rerank on before pinning the faithful scoring claim). |
| F3 | `src/popoe/render_rerank.py:236-239` + `scripts/sar_render_compare.py:247-265` (`PoseRenderer.calibrate`) | **One-shot y_sign calibration latched on a process-shared renderer, from the first champion pose, with the IoU verdict discarded.** `_calibrated` is set after calibrating against whatever pose first reaches `_sar_ti`; no IoU floor, no log line, never re-checked per asset/dataset. | The first candidate of the run is a degenerate RANSAC pose (the runner tolerates those by design); both signs project ~0 IoU and the argmax latches the wrong y_sign. Every subsequent `sar_ti` in the 20h formal run is computed on vertically mirrored renders — the exact failure `calibrate()` exists to prevent — turning every rerank decision into noise, silently. A clean smoke proves nothing: the full run is a new process with its own latch. | **must-disclose** (cheap hardening: log `(sign, iou)`, assert an IoU floor, or calibrate per asset). |
| F4 | `src/popoe/render_rerank.py:321-333` | **Re-ICP failure keeps the flip but the champion's s_icp.** If `icp_refinement` throws, the returned hypothesis has the flipped `R` with the pre-flip pose's `s_icp`/translation; ChampionScorer then multiplies a stale fitness with `s_feat_1` at the new pose. The only marker is `breakdown["render_rerank_re_icp"]="failed:…"`, which no CSV column carries. | An Open3D error on one candidate silently promotes a mis-scored flipped pose into the champion race for that target. | minor (rare exception path; safer to fall back to the champion variant on failure). |
| F5 | `src/popoe/render_rerank.py:266-271` (`_tau_icp` fallback) | Fallback recomputes `TAU_FRAC * extent` when `tau_icp` is absent from the breakdown. Unreachable today (ICPRefiner always writes it, `src/popoe/adapters.py:90` — the fix for the original archetype), but any future chain placing the reranker without/before ICPRefiner in a `--tau-diameter` run silently reintroduces the extent-vs-diameter basis switch. | Latent only. | minor (guard note). |
| F6 | `src/popoe/cache.py:94-98,107-111` | Concurrent writers to the same key share one deterministic tmp path (`<stage>_<key>.tmp.npz`); two parallel session lanes writing the same entry can interleave and publish a corrupt npz. Failure is loud on read (np.load raises → note_failure / startup crash), so not silent-wrong-answer, but it voids candidates mid-run. | Two pods/lanes warming one shared cache dir. | minor (use `tempfile` per-writer names). |
| F7 | `src/popoe/scoring.py:112` vs `src/popoe/freeze/adapters.py:173` | `ChampionScorer` defaults a missing `s_icp` to **0.0** (FreeZeScorer uses 1.0): a chain without ICPRefiner zeroes every score and the run completes as clean-looking "object not found" rows. Held safe today only by convention (formal chain always has ICP). | Latent only. | minor. |

## Checked and found clean (archetype-specific)

* `tau_icp` is now written by ICPRefiner (`adapters.py:90`) and reused by the
  reranker's re-ICP — the original 0.03-as-30mm defect is genuinely fixed on the
  formal chain, including under `--tau-diameter`.
* `--tau-diameter` plumbing: one basis feeds RANSAC tau, ICP tau and the
  scorer's `tau_abs` (`recipes.py:330-348`); `ChampionScorer` only recomputes
  extent-based tau when `tau_abs is None` (historical path). Consistent.
* `--corr-topk 10`: precomputed pairs are `[query_idx, target_idx]`, row-major
  aligned with `np.repeat` target ids, indices consistent with the subsampled
  clouds on restarts (`open3d_ransac.py:120-134`). Correct orientation for
  `registration_ransac_based_on_correspondence(pcd_q, pcd_t, …)`.
* `--seed 42`: `o3d.utility.random.seed(seed + restart)` per solve/restart;
  seed deliberately outside the feature cache key (features are solver-free) —
  documented and correct.
* enc_cfg cache-key completeness vs the faithful pins: `canon_basis`,
  `n_points` (POPOE_QUERY_POINTS), `n_views`, `query_canon`, `query_fill`,
  `query_min_views`, `query_views`, `grid` (effective env value), render
  backend, per-mesh shading parts — all reach the key. `POPOE_MESH_SHADING`
  is deliberately excluded (documented pre-fix-key A/B aliasing) and unset by
  every recipe. No missing key found besides F1's value-vs-key mismatch.
* `Detection.bbox` is converted xywh→xyxy at ingestion
  (`segmentor_detections.py:414`), matching the reranker's x0,y0,x1,y1 crop.
* PCA basis discipline: `install_pca(None)` refused; incomplete query cache
  entries fatal; PCA sign canonicalisation in fusion; identity-reduction marker
  distinct from None. All the known cross-basis doors are shut.
* `scale_vis` split: formal runs are geo-matched (vis_split None → true half);
  non-default `POPOE_VIS_DIM` is keyed and threaded explicitly; the sweep's
  w=1.0 fast-path vs reweighted float64 copies differ only in dtype (values
  exact under the cast).
* mm/m: GT `t/1000`, depth `*depth_scale/1000`, CSV `t*1000`, reranker
  `t_m*1000` into the mm renderer — all conversions on the right side.
* Resume semantics (row-count invariant, partial-target drop, legacy `time`
  column) and the fresh-`--out` guards in the recipe blocks — consistent.
* Dense ICP cloud: rebuilt per call with `rng(0)`, deterministic per mask,
  shared by main ICP and rerank re-ICP (same array), capped 3000 — symmetric
  between all compared variants after PR #29.

## Notes for the freeze decision

1. F1 is the only finding that corrupts *cached features* — before pinning,
   either fix the hit-path canon frame or mandate strictly cold caches for the
   faithful arms and forbid resume (both are operationally unrealistic for
   20h runs; fix the code).
2. F2/F3 do not corrupt caches; they can be disclosed or hardened without
   invalidating warm feature caches (both live in the pose/scoring path, same
   cache-safety argument as PR #29).
