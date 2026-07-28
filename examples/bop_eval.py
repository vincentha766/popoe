"""BOP evaluation through popoe stages — the framework-validation run.

Reproduces the reproduction study's formal pipeline semantics using popoe's
stage contracts composed at the runner level:

  detections Segmentor (label pooling) -> encode once at w=1 (pinned)
  -> for each (mask, visual weight): solve -> refine -> ChampionScorer
  -> per-mask champion, top inst_count champions -> one BOP CSV row each
     (inst_count==1, i.e. all of LMO/YCB-V, reproduces the old single-row
     global argmax exactly)

Workflow features (the experiment accelerators):

  * ``--cache DIR``  — per-candidate target features persist to disk; reruns
    that only change registration/scoring skip GeDi+DINO entirely.
  * ``--cand-csv F`` — every (mask x weight) hypothesis is dumped with its
    score breakdown, so selection rules can be swapped OFFLINE (zero GPU).
  * Resumable: rows already in --out are skipped on relaunch.

Dataset layout is resolved from the dataset name (default: the --bop
basename) via ``popoe.datasets.bop.BOP_LAYOUTS`` — split dir (tless/hb:
test_primesense), image modality (itodd: gray/*.tif), models dir (tless:
models_cad) — with --split / --models-dir overrides. An unknown name is
fatal, never guessed.

The ``time`` column holds the RAW per-target wall time. A BOP submission
requires one shared per-image total instead — run
``examples/bop_time_normalize.py`` on the CSV before uploading.

Per-object visual PCA is snapshotted at query encoding and re-installed
before each target encode (install_pca) — the image-major loop interleaves
objects, and the shared fusion instance would otherwise leak one object's
PCA into another's target features.

Usage (pod):
  python examples/bop_eval.py --bop /workspace/bop_data/ycbv \
      --detections /workspace/bop_data/detections/cnos/BOP23/fastSAM_pbr/fastSAM_pbr_ycbv.json \
      --out popoe_ycbv.csv --cache /workspace/popoe_cache_ycbv \
      [--objs 5,8,...] [--weights 1.0,0.7,0.5,0.3,0.2] [--cand-csv cands.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from popoe.adapters import (BestScoreSelector, resolve_resume,
                            select_top_instances)
from popoe.cache import StageCache, file_fingerprint, fingerprint
from popoe.datasets.bop import bop_layout, default_targets_path
from popoe.interfaces import ObjectModel, PointFeatures, PoseHypothesis, Scene
from popoe.confusable_select import dual_assign_hyps, partner_id
from popoe.freeze.feature_extractor import mesh_shading_key_parts
from popoe.freeze.recipes import (
    TAU_FRAC, WEIGHTS, YCBV_CLAMP_DIAMETERS_M, YCBV_MERGE_LABELS,
    best_encoders, best_segmentor, scale_vis, solver_provenance,
    stages_for_object,
)

IDN = " ".join(f"{v:.6f}" for v in np.eye(3).flatten())
ZT = "0.0 0.0 0.0"


def _parse_sources_arg(sources):
    return [s.strip() for s in sources.split(",") if s.strip()]


def validate_source_args(detections, sources):
    """Enforce the --detections / --sources one-of rule. Called EARLY (right
    after parse_args) so an argument error can never fire after the resume-
    cleanup step has already rewritten --out."""
    if bool(detections) == bool(sources):
        raise SystemExit("pass exactly one of --detections or --sources")
    if sources and not _parse_sources_arg(sources):
        raise SystemExit("--sources is empty")


def cand_csv_header(score_coarse):
    """Column header for the --cand-csv dump. The optional --score-coarse block
    (s_coarse plus the PRE-ICP pose it was measured at, R_coarse / t_coarse) is
    appended, then a `solver` column that stamps which pose solver produced each
    row (so a dump self-identifies its solver, and the header pins the mode on
    append).

    R_coarse / t_coarse ride along with s_coarse because they come from the same
    switch (ICPRefiner(keep_coarse=True)) and answer the question s_coarse only
    half answers: s_coarse says how good the pre-ICP pose LOOKED to the scorer,
    the pose itself lets it be scored against ground truth — i.e. how many AR
    points ICP actually moved. Same units and formatting as R / t (rotation
    row-major, translation in mm), so the same downstream reader handles both."""
    return (["scene_id", "im_id", "obj_id", "cand", "w", "s_icp", "s_feat_1",
             "metric_fit", "score", "R", "t"]
            + (["s_coarse", "R_coarse", "t_coarse"] if score_coarse else [])
            + ["solver"])



def probe_corr_stats(q, tgt, gts, syms, diam_m):
    """Of the correspondences the matcher would hand RANSAC, how many are
    geometrically right under GT?

      rate1   inlier fraction of the top-1 set (the k=1 path's whole pool)
      rate10  inlier fraction of the top-10 set (the corr_topk=10 pool)
      reach10 targets whose true match is anywhere in their 10 - the
              ceiling any sampler over that pool could reach

    tau = 3% of the object diameter (the paper's basis). Distances are the
    min over GT instances AND symmetry transforms, else a perfectly matched
    symmetric object would score 0%. Features are the cached w=1 canonical
    space - the same space s_feat_1 scores in."""
    qn = q.feats / np.maximum(
        np.linalg.norm(q.feats, axis=1, keepdims=True), 1e-12)
    tn = tgt.feats / np.maximum(
        np.linalg.norm(tgt.feats, axis=1, keepdims=True), 1e-12)
    sim = tn @ qn.T                                   # (Nt, Nq)
    k = min(10, sim.shape[1])
    top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    row = np.arange(sim.shape[0])[:, None]
    top = top[row, np.argsort(-sim[row, top], axis=1)]  # col 0 = argmax
    tau = 0.03 * diam_m
    best = np.full((sim.shape[0], k), np.inf)
    for g in gts:
        Rg, tg = g["R"], g["t"] / 1000.0              # GT t is mm
        for sym in syms:
            Rs = np.asarray(sym["R"], dtype=float)
            ts = np.asarray(sym["t"], dtype=float).reshape(3) / 1000.0
            posed = (q.pts @ Rs.T + ts) @ Rg.T + tg   # (Nq, 3), metres
            d = np.linalg.norm(posed[top] - tgt.pts[:, None, :], axis=2)
            best = np.minimum(best, d)
    return (float((best[:, 0] < tau).mean()),
            float((best < tau).mean()),
            float((best < tau).any(axis=1).mean()),
            float(np.median(best[:, 0]) * 1000.0),
            tau * 1000.0)

def resolve_layout(bop: Path, dataset=None, split=None, models_dir=None):
    """Dataset name + directory layout for --bop, name defaulting to the --bop
    basename. Both feed correctness-relevant decisions (split dir, image
    modality, models dir, the 'auto' merge), so an unrecognised name is fatal
    rather than defaulted: the rgb/png guess on itodd would not crash — it
    would complete the whole dataset as a clean-looking all-zero CSV."""
    name = (dataset or bop.name).lower()
    try:
        return name, bop_layout(name, split=split, models_dir=models_dir)
    except ValueError as e:
        # Two ways to get here deserve two hints: telling someone who already
        # passed --dataset to "pass --dataset explicitly" answers nothing.
        hint = (f"--dataset {dataset!r} is not a known dataset" if dataset
                else f"--bop basename {bop.name!r} does not name a dataset; "
                     f"pass --dataset explicitly")
        raise SystemExit(
            f"{e}\n{hint} (layout and the 'auto' merge depend on it).")


def resolve_merge(merge_arg, dataset):
    """Resolve --merge. 'auto' (the default) applies the YCB-V clamp pooling
    on ycbv and nothing elsewhere: obj ids 19/20 also exist on tless, itodd
    and hb, where the former literal 'ycbv' default silently pooled two
    unrelated objects — and handed them the clamp-specific size-aware scorer
    (stages_for_object's `size_aware=obj_id in merge`)."""
    if merge_arg == "auto":
        return dict(YCBV_MERGE_LABELS) if dataset == "ycbv" else {}
    if merge_arg == "ycbv":
        return dict(YCBV_MERGE_LABELS)
    if merge_arg == "none":
        return {}
    merge = {}
    for grp in merge_arg.split(","):
        ids = [int(x) for x in grp.split(":")]
        for a in ids:
            merge[a] = ids
    return merge


def dense_mask_cloud(scene, mask, max_pts: int = 3000):
    """P_T^dense (FreeZeV2 Eq. 6): every valid depth pixel inside the mask,
    back-projected with the scene intrinsics.

    Byte-for-byte the same construction the feature extractor already performs
    for the GeDi neighbourhood (`pcd_dense` there, `(depth > 0) & mask`, metres,
    camera frame) — it is rebuilt here because that one is discarded before it
    can reach ICP, and recomputing it is cheaper than plumbing it through the
    feature cache (and would have invalidated every cached entry).

    `max_pts` caps the result with a fixed-seed uniform draw (0 = no cap). The
    draw is seeded per call, so the same mask always yields the same cloud and
    the run stays reproducible.

    Returns None when the mask has fewer than 4 valid depth pixels; the caller
    then leaves `pts_dense` unset and ICP falls back to the sparse cloud."""
    ys, xs = np.where((scene.depth > 0) & mask)
    if len(ys) < 4:
        return None
    if max_pts and len(ys) > max_pts:
        idx = np.sort(np.random.default_rng(0).choice(len(ys), max_pts,
                                                      replace=False))
        ys, xs = ys[idx], xs[idx]
    d = scene.depth[ys, xs]
    fx, fy = scene.K[0, 0], scene.K[1, 1]
    cx, cy = scene.K[0, 2], scene.K[1, 2]
    return np.stack([(xs - cx) * d / fx, (ys - cy) * d / fy, d],
                    axis=1).astype(np.float32)


def read_frame_images(sdir: Path, im_id: int, layout):
    """Read one image's (bgr, raw depth), or raise FileNotFoundError naming
    the file (cv2.imread returns None for missing and unreadable files alike,
    so the None check must sit this close to the read — the message says so).

    A gray modality (itodd) is read UNCHANGED and replicated to 3 channels
    explicitly. IMREAD_COLOR would do the replication too, but it also
    silently shifts a 16-bit image down to its top 8 bits — near-black input
    reaching the visual branch with no error anywhere. Until someone chooses
    a tone mapping deliberately, a non-uint8 gray image is fatal."""
    depth_path = sdir / "depth" / f"{im_id:06d}{layout['depth_ext']}"
    img_path = sdir / layout["img_dir"] / f"{im_id:06d}{layout['img_ext']}"
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    gray = layout["img_dir"] == "gray"
    img = cv2.imread(str(img_path),
                     cv2.IMREAD_UNCHANGED if gray else cv2.IMREAD_COLOR)
    if depth_raw is None or img is None:
        missing = depth_path if depth_raw is None else img_path
        raise FileNotFoundError(str(missing))
    if gray:
        if img.dtype != np.uint8:
            raise SystemExit(
                f"{img_path} decodes to {img.dtype}, not uint8 — a 16-bit "
                f"gray image needs a deliberate tone-mapping choice; refusing "
                f"to shift to the top 8 bits implicitly.")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img, depth_raw


def floored_topk(user_topk, max_inst):
    """Detection top-K must not cap output below the largest inst_count, or a
    4-instance target could never receive 4 champions. For a multi-source union
    this floor holds PER (source, label) bucket (see BOPDetectionsSegmentor), so
    every backend can still supply enough candidates for a multi-instance
    target — it is not a shared global cap one source could exhaust."""
    return max(user_topk, max_inst)


def load_cached_query(cache, qkey):
    """Return a cached query entry's arrays, or None for a clean miss.

    A query entry is BOTH files: the arrays and the PCA sidecar whose basis
    their visual half lives in. Half an entry is not a hit.

    An INCOMPLETE entry (arrays present, sidecar missing) is fatal, not
    repairable. Re-encoding the query would restore a correct basis, but the
    target key is `fingerprint("target", ..., qkey)` and `qkey` is derived from
    config + mesh content only — so it does not change, and every target entry
    written earlier stays reachable under it. Those targets may have been
    encoded by the pre-fix code that let the target side fit its OWN basis
    (see FreeZeTargetEncoder.install_pca), and nothing in a cached entry says
    which basis produced it. Repairing the query alone would therefore leave
    poisoned targets matched against a corrected query — the same silent
    cross-basis comparison, now harder to spot.

    Under the current write order (sidecar first, arrays second) a crash can
    only ever leave an ORPHAN SIDECAR, never this state. So arrays-without-
    sidecar means a pre-fix cache or a hand-deleted `.pkl`, and in both cases
    the dependent targets are suspect and unidentifiable. Fail fast — this runs
    in the up-front query loop, before any real work.
    """
    if cache is None or qkey is None:
        return None
    arrays = cache.get_arrays("query", qkey)
    if arrays is None:
        return None
    pca_vis = cache.get_pickle("query", qkey)
    if pca_vis is None:
        raise SystemExit(
            f"query cache entry {qkey} is INCOMPLETE: arrays present, PCA "
            f"sidecar missing.\n"
            f"This cannot be repaired in place. Target entries are keyed by "
            f"this same qkey, so re-encoding the query would not invalidate "
            f"them, and any of them may hold features in a target-fitted "
            f"basis written by the pre-fix code — cached features do not "
            f"record which basis produced them.\n"
            f"Use a fresh --cache directory (or delete this one) and re-run. "
            f"Under the current write order this state cannot arise from a "
            f"crash, so the cache predates the fix or a .pkl was deleted.")
    return arrays, pca_vis


def resolve_segmentor(detections, sources, topk, merge_labels,
                      size_select=None, confusable_diameters=None,
                      size_select_fallback=True):
    """Build the detections segmentor from the mutually-exclusive --detections /
    --sources knobs (exactly one required).

    --sources is a comma-separated ``name=path`` list unioned as named backends
    (e.g. ``cnos=/a.json,nids=/b.json``). ``topk`` is the per-(source, label)
    bucket cap and is passed straight through (already floored by the caller).

    ``size_select`` is opt-in (default None = formal path). When set, pass
    confusable diameters for the merged pair (YCB-V clamps)."""
    validate_source_args(detections, sources)
    kw = dict(
        topk=topk,
        merge_labels=merge_labels,
        size_select=size_select,
        confusable_diameters=confusable_diameters,
        size_select_fallback=size_select_fallback,
    )
    if sources:
        return best_segmentor(sources=_parse_sources_arg(sources), **kw)
    return best_segmentor(detections, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bop", required=True)
    ap.add_argument("--dataset", default=None,
                    help="BOP dataset name (lmo|tless|tudl|icbin|itodd|hb|"
                         "ycbv). Default: the basename of --bop. Drives the "
                         "split dir, image modality/extension, models dir and "
                         "the 'auto' merge default; an unknown name is fatal, "
                         "never guessed.")
    ap.add_argument("--split", default=None,
                    help="override the split dir from the dataset layout "
                         "(e.g. test_kinect on hb; tless/hb default to "
                         "test_primesense)")
    ap.add_argument("--models-dir", default=None,
                    help="override the models dir from the dataset layout "
                         "(tless defaults to models_cad — it has no models/)")
    ap.add_argument("--allow-missing-images", action="store_true",
                    help="write zero rows (counted as load_image failures in "
                         "the end-of-run summary) for images whose image/depth "
                         "files cannot be read, instead of aborting. Default "
                         "aborts: under a wrong layout every image is missing "
                         "and the run would complete as a clean-looking "
                         "all-zero CSV.")
    ap.add_argument("--detections", default=None,
                    help="single BOP detections JSON. Mutually exclusive with "
                         "--sources; pass exactly one.")
    ap.add_argument("--sources", default="",
                    help="comma-separated name=path list unioned as named "
                         "backends, e.g. cnos=/a.json,nids=/b.json. Mutually "
                         "exclusive with --detections.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--objs", default="")
    ap.add_argument("--weights", default=",".join(str(w) for w in WEIGHTS))
    ap.add_argument("--topk", type=int, default=2,
                    help="detections kept per (source,label) bucket; floored "
                         "at the dataset's max inst_count so multi-instance "
                         "targets can receive enough champions. With --sources "
                         "this cap is PER source (each backend supplies up to "
                         "topk per label), not a shared global cap")
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--cache", default="", help="target-feature cache dir")
    ap.add_argument("--cand-csv", default="", help="dump every hypothesis")
    ap.add_argument("--score-coarse", action="store_true",
                    help="also record the paper's S_coarse (pre-ICP feature "
                         "score, canonical w=1) as an s_coarse column in "
                         "--cand-csv. Diagnostic only: the champion score/R/t "
                         "(and the main --out rows) are unchanged; only the "
                         "extra s_coarse column and the wall-clock `time` "
                         "column (more compute) differ.")
    ap.add_argument("--use-s-coarse", action="store_true",
                    help="USE S_coarse in arbitration: score *= max(s_coarse,0) "
                         "(rule s_icp*s_feat_1*metric_fit*s_coarse). Per-DATASET "
                         "switch — helps YCB-V, hurts LM-O. Implies "
                         "--score-coarse (records the s_coarse column too). This "
                         "DOES change the main --out score/R/t, so use a FRESH "
                         "--out and --cand-csv: resume/append are keyed by rows, "
                         "not config, so reusing a file written without this "
                         "flag silently keeps its old scores (as with any "
                         "score-affecting knob — --merge, --weights, --grid).")
    ap.add_argument("--solver", default="o3d",
                    choices=["o3d", "gpu", "gpu-feat", "teaser"],
                    help="pose solver: o3d (default, evaluated mainline — "
                         "unchanged) | gpu (ported batched RANSAC, geometric "
                         "fitness) | gpu-feat (gpu with the Eq.5 feature-aware "
                         "fitness, the B layer) | teaser (TEASER++ certifiable "
                         "registration). gpu* need torch; teaser needs "
                         "teaserpp_python (source build). A non-default "
                         "solver changes score/R/t, so use a FRESH --out and "
                         "--cand-csv: resume/append are keyed by rows, not "
                         "config (as with --use-s-coarse/--merge/--weights).")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed the solver's RNG. Default None = the historical "
                         "UNSEEDED mainline: Open3D's RANSAC draws from a global "
                         "RNG it never seeds, so --solver o3d runs are not "
                         "reproducible run-to-run. Pass a seed for anything whose "
                         "numbers get cited — REPRODUCTION.md's acceptance rule "
                         "tightens to bit-identical for seeded runs. Seeding "
                         "changes score/R/t, so use a FRESH --out and --cand-csv. "
                         "Ignored by --solver teaser (no RNG); overrides the gpu "
                         "solvers' own default 42. Feature caches are NOT affected "
                         "— the seed moves poses, not features, so it is "
                         "deliberately absent from the encoder cache key.")
    ap.add_argument("--merge", default="auto",
                    help="'auto' (default: the ycbv clamp pooling on ycbv, "
                         "none elsewhere — obj ids 19/20 exist on tless/itodd/"
                         "hb too and must not be pooled there), 'ycbv', "
                         "'none', or '19:20,...'")
    ap.add_argument("--size-select", default="none",
                    choices=["none", "soft", "nearest"],
                    help="opt-in mask-stage size arbitration for confusable "
                         "pairs (YCB-V clamps). 'none' (default) is the formal "
                         "headline path; 'nearest'/'soft' re-rank pooled masks "
                         "by depth extent vs CAD diameters (needs scene depth). "
                         "Score-affecting — use a FRESH --out / --cand-csv.")
    ap.add_argument("--size-select-no-fallback", action="store_true",
                    help="with --size-select nearest: do not fall back to pure "
                         "appearance when no mask matches the query diameter")
    ap.add_argument("--dual-assign", action="store_true",
                    help="for confusable merge pairs co-visible in one image, "
                         "assign each cand index to the CAD with higher "
                         "metric_fit before writing the champion (multi-object "
                         "dual-CAD). Requires --merge with a pair (e.g. ycbv). "
                         "Score-affecting — use a FRESH --out / --cand-csv. "
                         "YCB-V inst_count==1 only.")
    ap.add_argument("--icp-dense", action="store_true",
                    help="FIDELITY FIX (FreeZeV2 Eq. 6): refine against the "
                         "DENSE target cloud — every valid depth pixel inside "
                         "the mask — instead of the 16x16/32x32 patch-grid "
                         "cloud. The grid cloud's own point spacing (~2.5 mm "
                         "median) is the same order as tau_ICP (3-7 mm), so ICP "
                         "cannot resolve below it. Pose-side only: absent from "
                         "the encoder cache key, so feature caches still hit. "
                         "Score-affecting (s_icp rises) — use a FRESH --out.")
    ap.add_argument("--icp-dense-max", type=int, default=3000,
                    help="cap on |P_T^dense| for --icp-dense (0 = no cap). "
                         "Default 3000 is the paper's own budget (Sec. IV-A: "
                         "'For P_Q and P_T^dense we use 5k and 3k points, "
                         "respectively, while P_T^sparse includes up to 256 "
                         "points'). Subsampling is a fixed-seed uniform draw. "
                         "Two of those three numbers are still unmatched here: "
                         "P_Q is 3000 against the paper's 5k, and --grid 32 "
                         "gives P_T^sparse up to 1024 against its 256.")
    ap.add_argument("--n-restarts", type=int, default=1,
                    help="o3d solver restarts per (mask, weight); restart>0 "
                         "runs on a deterministic 70%% subsample and every "
                         "restart's pose becomes a separate hypothesis. The "
                         "weight sweep gives each mask 5 solve attempts in 5 "
                         "feature spaces; this knob gives it N attempts in ONE "
                         "space, which is the control that separates "
                         "'diversity of feature spaces' from 'more RANSAC "
                         "budget' in the sweep's measured value.")
    ap.add_argument("--probe-corr", default="",
                    help="CORRESPONDENCE PROBE: instead of solving, measure "
                         "how many of the correspondences the matcher would "
                         "hand RANSAC are geometrically right under GT (w=1 "
                         "canonical features; tau = 3%% of the diameter). "
                         "Writes one row per (target, mask) to this CSV. "
                         "Cache-only and CPU-capable: encoders never load, "
                         "every cache miss is fatal or counted. Feature "
                         "quality gets measured WITHOUT RANSAC in the way, "
                         "which is the point.")
    ap.add_argument("--corr-topk", type=int, default=0,
                    help="FIDELITY KNOB (FreeZeV2 Sec. IV-A: 'a top-k strategy "
                         "with k = 10'): precompute, for each target point, its "
                         "k best query matches and hand RANSAC that set, "
                         "instead of Open3D's internal k=1 matching. 0 = "
                         "historical path, byte-identical. o3d solver only.")
    ap.add_argument("--tau-diameter", action="store_true",
                    help="FIDELITY FIX (FreeZeV2 Sec. IV-A): set tau_inlier / "
                         "tau_ICP / the feature-score inlier radius to 3% of "
                         "the BOP models_info DIAMETER, not 3% of the sampled "
                         "query cloud's largest bounding-box side (which is "
                         "2-35% smaller on LM-O). Pose-side only; caches hit. "
                         "Score-affecting — use a FRESH --out.")
    ap.add_argument("--render-backend", default="nvdiffrast",
                    choices=["nvdiffrast", "trimesh", "auto"],
                    help="CAD renderer for query features. Default demands the "
                         "GPU rasteriser (what the reported numbers used) and "
                         "errors without it; 'auto' accepts the ~100x slower CPU "
                         "ray-caster, which yields DIFFERENT features.")
    args = ap.parse_args()
    # --use-s-coarse implies recording it too (the s_coarse cand-csv column and
    # the coarse-pose wiring), so it depends on --score-coarse.
    if args.use_s_coarse:
        args.score_coarse = True
    # Validate the source-mode BEFORE any file work (resume cleanup rewrites
    # --out): a CLI-argument error must not fire after destructive steps.
    validate_source_args(args.detections, args.sources)
    if args.probe_corr:
        # Fail-fast guards, BEFORE any filesystem work. CPU-capable by
        # construction: every feature comes from the cache, so the heavy
        # encoders never load. 'auto' would make the render_backend key part
        # describe THIS box's GPU rather than the box that built the cache —
        # demand the explicit value the cache was built with.
        if args.render_backend == "auto":
            raise SystemExit("--probe-corr needs an explicit --render-backend "
                             "(the value the cache was built with; the "
                             "campaign caches say nvdiffrast).")
        if not args.cache:
            raise SystemExit("--probe-corr reads features, it does not create "
                             "them; pass --cache.")

    bop = Path(args.bop)
    ds_name, layout = resolve_layout(bop, args.dataset, args.split,
                                     args.models_dir)
    # Provenance line for the layout: like the solver line below, what the run
    # resolved belongs in the log next to the numbers it produced.
    print(f"dataset={ds_name} split={layout['split']} "
          f"images={layout['img_dir']}/*{layout['img_ext']} "
          f"models={layout['models_dir']}", flush=True)
    weights = [float(w) for w in args.weights.split(",")]
    obj_filter = {int(x) for x in args.objs.split(",") if x.strip()}
    merge = resolve_merge(args.merge, ds_name)
    if args.merge == "auto":
        print("--merge auto -> "
              + ("ycbv clamp pooling" if merge else "none"), flush=True)
    size_select = None if args.size_select == "none" else args.size_select
    confusable_diameters = None
    if size_select is not None:
        # Diameters for every id that participates in a merge group; fall back
        # to the known YCB-V clamp table (lab recipe).
        confusable_diameters = dict(YCBV_CLAMP_DIAMETERS_M)
        if merge:
            for ids in merge.values():
                for oid in ids:
                    confusable_diameters.setdefault(
                        int(oid), YCBV_CLAMP_DIAMETERS_M.get(int(oid), 0.0))
            confusable_diameters = {
                k: v for k, v in confusable_diameters.items() if v > 0
            }
        if not confusable_diameters:
            raise SystemExit(
                "--size-select needs confusable diameters; use --merge ycbv "
                "or extend YCBV_CLAMP_DIAMETERS_M")
    if args.cache:
        os.makedirs(args.cache, exist_ok=True)

    targets = json.load(open(default_targets_path(bop, layout["split"])))
    if obj_filter:
        targets = [t for t in targets if t["obj_id"] in obj_filter]
    by_img: dict = {}
    for t in targets:
        # inst_count: how many instances of this object the image contains.
        # Ignoring it (the old code) caps recall at one instance per object —
        # invisible on LMO/YCB-V (always 1), wrong on multi-instance datasets.
        by_img.setdefault((t["scene_id"], t["im_id"]), []).append(
            (t["obj_id"], int(t.get("inst_count", 1))))

    # Resume, by ROW COUNT alone: the writer's completion invariant is that a
    # finished target has EXACTLY inst_count rows (zero-padded when fewer
    # champions were found), so fewer rows == crash mid-target. Those stale
    # rows are dropped before re-running — appending would duplicate
    # instances. Row CONTENTS are deliberately not consulted (a real score can
    # format as 0.000000). With inst_count==1 everywhere this reduces exactly
    # to the old any-row rule.
    target_counts = {(s, i, o): n
                     for (s, i), objs in by_img.items() for o, n in objs}
    done: set = set()
    if os.path.exists(args.out):
        row_stats: dict = {}
        for r in csv.DictReader(open(args.out)):
            key = (int(r["scene_id"]), int(r["im_id"]), int(r["obj_id"]))
            row_stats[key] = row_stats.get(key, 0) + 1
        done, partial = resolve_resume(row_stats, target_counts)
        if partial:
            print(f"dropping {len(partial)} partially-written multi-instance "
                  f"target(s) from {args.out} for re-run", flush=True)
            with open(args.out) as fin:
                rows = list(csv.DictReader(fin))
            tmp = args.out + ".tmp"
            with open(tmp, "w", newline="") as fout:
                wr0 = csv.writer(fout)
                wr0.writerow(["scene_id", "im_id", "obj_id", "score", "R", "t",
                              "time"])
                for r in rows:
                    key = (int(r["scene_id"]), int(r["im_id"]),
                           int(r["obj_id"]))
                    if key not in partial:
                        # .get: a legacy CSV without the time column must not
                        # KeyError the cleanup rewrite.
                        wr0.writerow([r["scene_id"], r["im_id"], r["obj_id"],
                                      r["score"], r["R"], r["t"],
                                      r.get("time", "")])
            os.replace(tmp, args.out)
    print(f"{len(targets)} targets / {len(by_img)} images"
          + (f" (resuming past {len(done)})" if done else ""), flush=True)

    # Detection top-K floored at the dataset's max inst_count (see floored_topk);
    # for a union the floor holds per (source, label) bucket.
    max_inst = max(target_counts.values(), default=1)
    segmentor = resolve_segmentor(
        args.detections, args.sources,
        topk=floored_topk(args.topk, max_inst),
        merge_labels=merge,
        size_select=size_select,
        confusable_diameters=confusable_diameters,
        size_select_fallback=not args.size_select_no_fallback,
    )
    if size_select:
        print(f"size_select={size_select} diameters={confusable_diameters}",
              flush=True)
    if args.dual_assign:
        if not merge:
            raise SystemExit("--dual-assign requires --merge with a confusable pair")
        print(f"dual_assign=ON merge={merge}", flush=True)
    # Provenance: reproducibility is a property of the RUN, so it belongs in the
    # log next to the numbers, not only in the shell history that produced them.
    # Built by recipes so it reports the EFFECTIVE seed per solver family — the
    # gpu solvers are deterministic by default and teaser has no RNG, so a flat
    # "UNSEEDED" would be a false claim in a cited run's log.
    print(solver_provenance(args.solver, args.seed), flush=True)
    if args.probe_corr:
        q_enc = t_enc = None    # guards ran at arg-validation time
    else:
        q_enc, t_enc = best_encoders(target_grid=args.grid,
                                     render_backend=args.render_backend)
    selector = BestScoreSelector()

    # Config-addressed stage cache: keys fingerprint the encoder configuration
    # and input CONTENT (mesh bytes, mask pixels) plus — for targets — the
    # query key whose PCA fit defines their basis. Same config -> automatic
    # reuse; any knob change invalidates exactly what it should. See
    # popoe/cache.py for the two invariants this encodes.
    enc_cfg = {
        # Effective value, not args.grid: best_encoders() only setdefault()s the
        # env var, so a pre-set POPOE_TARGET_GRID wins over --grid and the key
        # must record what actually ran.
        "grid": os.environ.get("POPOE_TARGET_GRID", str(args.grid)),
        # Effective |P_Q| (POPOE_QUERY_POINTS reaches best_encoders). int, and
        # 3000 at the default, so every pre-existing key is byte-identical.
        "n_points": int(os.environ.get("POPOE_QUERY_POINTS", "3000")),
        "dino_layer": os.environ.get("POPOE_DINO_LAYER", "ratio0.78"),
        "two_scale": os.environ.get("POPOE_TWO_SCALE_GEDI", "1"),
        "crop": os.environ.get("POPOE_TARGET_CROP", "1"),
        "vis_dim": os.environ.get("POPOE_VIS_DIM", "geo-matched"),
        # Every remaining env knob that changes the features. Defaults mirror
        # the reads in feature_extractor.py / fusion.py; if you add a knob
        # there, add it here or cached features will survive the change.
        "n_views": os.environ.get("POPOE_N_VIEWS", "162"),
        "target_fill": os.environ.get("POPOE_TARGET_FILL", "0.5"),
        "target_canon": os.environ.get("POPOE_TARGET_CANON", "224"),
        # Extraction weight is PINNED by best_encoders (w=1); the env var no
        # longer reaches these encoders. Record the pin, not the env.
        "vis_weight": "1.0-pinned",
        "skip_vis": os.environ.get("POPOE_SKIP_VIS", "0"),
        "geom_backbone": os.environ.get("POPOE_GEOM_BACKBONE", "gedi"),
        "dgedi_mode": os.environ.get("POPOE_DGEDI_MODE", "single_scale"),
        "gedi_path": os.environ.get("POPOE_GEDI_PATH", "/workspace/gedi"),
        # The renderer is an upstream knob like any other: nvdiffrast and the
        # trimesh CPU ray-caster produce different CAD views, hence different
        # query features. It used to be absent from the key, so a cache built on
        # a box without nvdiffrast was silently reused on one with it.
        "render_backend": (args.render_backend if args.probe_corr
                           else q_enc.render_backend),
    }
    # .lower() to match how load_gedi() resolves the backbone: POPOE_GEOM_BACKBONE
    # =FPFH loads FPFH, and its knobs must reach the key or a radius sweep reuses
    # the previous radius' features.
    if enc_cfg["geom_backbone"].lower() == "fpfh":
        # Added ONLY when FPFH is the active backbone: unconditional keys would
        # change every existing GeDi key and throw away the pod-side cache.
        from popoe.descriptors import fpfh_config
        enc_cfg.update(fpfh_config())
    # Paper-fidelity feature knobs (FreeZeV2 Sec. IV-A). Same conditional-add
    # rule as FPFH above: each enters the key ONLY at a non-default value, so
    # every cache built before these knobs existed keeps its exact key — and a
    # faithful run can never be served the historical features by mistake.
    for _key, (_env, _dflt) in {
        "query_canon": ("POPOE_QUERY_CANON", "224"),      # render canvas px
        "query_fill": ("POPOE_QUERY_FILL", "0.45"),       # object fill fraction
        "query_min_views": ("POPOE_QUERY_MIN_VIEWS", "0"),  # visibility gate
        "canon_basis": ("POPOE_CANON_BASIS", "extent"),   # GeDi radius basis
    }.items():
        _val = os.environ.get(_env, _dflt)
        if _val != _dflt:
            enc_cfg[_key] = _val
    # Where the [vis | geo] boundary sits, for the selection-time weight sweep.
    # Taken from the SAME value the cache key records, so the split can never
    # disagree with the features it is applied to: a different POPOE_VIS_DIM is
    # a different key, hence a different entry. "geo-matched" (the default, and
    # what every published number ran under) means equal halves, which
    # scale_vis derives itself.
    vis_split = (None if enc_cfg["vis_dim"] == "geo-matched"
                 else int(enc_cfg["vis_dim"]))
    if vis_split is not None:
        print(f"vis_dim={vis_split} (NOT geo-matched — the fused halves are "
              f"unequal and the weight sweep splits at {vis_split})", flush=True)

    cache = StageCache(args.cache) if args.cache else None

    # --tau-diameter: the paper's threshold basis. Loaded from the SAME models
    # directory the meshes come from, so the diameter always describes the mesh
    # that was actually encoded. Read once, up front, so a missing/renamed
    # models_info.json fails before any GPU work.
    diameters_m: dict = {}
    if args.tau_diameter or args.probe_corr:
        mi_path = bop / layout["models_dir"] / "models_info.json"
        if not mi_path.exists():
            raise SystemExit(f"--tau-diameter/--probe-corr need {mi_path} "
                             f"(BOP ships it next to the meshes); not found.")
        diameters_m = {int(k): float(v["diameter"]) / 1000.0
                       for k, v in json.load(open(mi_path)).items()}

    # Encode ALL queries up front (fail fast; PCA snapshots live in meta).
    # Query features + fitted PCA are CACHED alongside target features: the
    # cached targets' visual halves are projected in the query PCA basis, so
    # re-fitting the PCA in a later run silently breaks them (PCA signs are
    # arbitrary per fit — see fusion.py). One basis per object, persisted.
    query_cache: dict = {}
    for obj_id in sorted({o for objs in by_img.values() for o, _n in objs}):
        obj = ObjectModel(obj_id=obj_id,
                          mesh_path=str(bop / layout["models_dir"]
                                        / f"obj_{obj_id:06d}.ply"),
                          diameter=0.0)
        t0 = time.time()
        # Shading is keyed PER MESH, not in enc_cfg: the fix that taught the
        # renderer about `property uchar red/green/blue` changed the rendered
        # pixels for LM-O / TUD-L / IC-BIN / HB only, and neither enc_cfg nor
        # the mesh bytes move when render CODE changes — so without this part
        # the old grey-render features would be served silently. Empty tuple for
        # UV-atlas and colourless meshes, which keeps every YCB-V / T-LESS /
        # ITODD key (and hence their cached features) exactly as it was.
        shade_parts = mesh_shading_key_parts(obj.mesh_path)
        qkey = (fingerprint("query", enc_cfg, file_fingerprint(obj.mesh_path),
                            obj_id, *shade_parts) if cache else None)
        # Both files or nothing; an incomplete entry is fatal (the dependent
        # target entries share this qkey and may hold a target-fitted basis).
        entry = load_cached_query(cache, qkey)
        if entry is not None:
            hit, pca_hit = entry
            q = PointFeatures(pts=hit["pts"], feats=hit["feats"],
                              meta={"pca_vis": pca_hit})
            from popoe.interfaces import CanonFrame
            q.meta["canon_frame"] = CanonFrame.from_points(q.pts)
        else:
            if args.probe_corr:
                raise SystemExit(
                    f"--probe-corr: query cache miss for obj {obj_id} — the "
                    f"probe reads features, it does not create them. Point "
                    f"--cache at the campaign cache and match its enc_cfg env "
                    f"knobs (grid, POPOE_QUERY_*, POPOE_CANON_BASIS, ...).")
            q = q_enc.encode_query(obj)
            if cache:
                # Sidecar FIRST, arrays second: the .npz is the commit marker,
                # so a crash between the two writes leaves an orphan .pkl (a
                # miss, harmless) rather than arrays with no basis (the silent
                # corruption above). Neither write is atomic with the other.
                cache.put_pickle("query", qkey, q.meta.get("pca_vis"))
                cache.put_arrays("query", qkey, pts=q.pts, feats=q.feats)
        q.meta["qkey"] = qkey
        q.meta["feats_w1"] = q.feats   # genuinely w=1: extraction is pinned
        extent = float(np.ptp(q.pts, axis=0).max())
        tau_basis = None
        if args.tau_diameter:
            if obj_id not in diameters_m:
                raise SystemExit(f"--tau-diameter: obj {obj_id} absent from "
                                 f"{layout['models_dir']}/models_info.json")
            tau_basis = diameters_m[obj_id]
        stages = stages_for_object(extent, size_aware=obj_id in merge,
                                   score_coarse=args.score_coarse,
                                   use_s_coarse=args.use_s_coarse,
                                   corr_topk=args.corr_topk,
                                   n_restarts=args.n_restarts,
                                   solver=args.solver, seed=args.seed,
                                   tau_basis_m=tau_basis)
        query_cache[obj_id] = (obj, q, stages)
        tau_note = ("" if tau_basis is None else
                    f" diam={tau_basis*1000:.0f}mm "
                    f"tau={TAU_FRAC*tau_basis*1000:.2f}mm "
                    f"(was {TAU_FRAC*extent*1000:.2f}mm)")
        # Both tags are printed unconditionally, including when they are empty
        # or default: this line is the evidence for "which renderer path and
        # which threshold basis actually ran". A log that only speaks up in the
        # interesting case cannot tell "flat, correctly" apart from "the
        # resolver never ran", and the same goes for the tau basis.
        shade_note = shade_parts[0].split("=")[1] if shade_parts else "uv-or-flat"
        print(f"  obj{obj_id}: extent={extent*1000:.0f}mm shading={shade_note} "
              f"encode={time.time()-t0:.1f}s{tau_note}", flush=True)

    cand_f = None
    if args.cand_csv:
        header = cand_csv_header(args.score_coarse)
        new = not os.path.exists(args.cand_csv)
        if not new:
            # Appending: the existing header must match this run's column set,
            # or --score-coarse toggling between runs would write rows under a
            # mismatched header (silent schema corruption).
            with open(args.cand_csv, newline="") as fchk:
                existing = next(csv.reader(fchk), [])
            if existing != header:
                raise SystemExit(
                    f"--cand-csv {args.cand_csv} has header {existing} but this "
                    f"run would write {header} (—score-coarse mismatch). Use a "
                    f"fresh file or match the flag.")
        cand_f = open(args.cand_csv, "a", newline="")
        cand_wr = csv.writer(cand_f)
        if new:
            cand_wr.writerow(header)

    # --probe-corr: GT, symmetries, output writer, and the stats function.
    probe_wr = None
    probe_gt_cache: dict = {}
    probe_syms: dict = {}
    if args.probe_corr:
        import sys as _sys
        _sys.path.insert(0, os.environ.get("POPOE_BOP_TOOLKIT",
                                           "/workspace/bop_toolkit"))
        from bop_toolkit_lib import misc as _btk_misc
        mi_raw = json.load(open(bop / layout["models_dir"] / "models_info.json"))
        for _k, _v in mi_raw.items():
            # Continuous symmetries discretised the way the AR metric does it;
            # without syms a perfectly matched sym object would read as 0%.
            probe_syms[int(_k)] = _btk_misc.get_symmetry_transformations(_v, 0.01)
        probe_f = open(args.probe_corr, "w", newline="")
        probe_wr = csv.writer(probe_f)
        probe_wr.writerow(["scene_id", "im_id", "obj_id", "cand", "n_t", "n_q",
                           "n_gt", "rate1", "rate10", "reach10", "med1_mm",
                           "tau_mm"])

    dense_sizes: list = []

    def encode_target_cached(scene, det, obj, q, ci, scene_fp):
        # scene_fp content-hashes rgb/depth/K (invariant 2 in cache.py): the
        # BOP ids alone would silently serve stale features if the files under
        # the same ids ever change (corrected dataset, different --bop root).
        key = (fingerprint("target", enc_cfg, scene_fp,
                           obj.obj_id, det.mask, q.meta["qkey"])
               if cache else None)
        hit = cache.get_arrays("target", key) if cache else None
        if hit is not None:
            tgt = PointFeatures(pts=hit["pts"], feats=hit["feats"],
                                meta={"feats_w1": hit["feats"]})
        else:
            t_enc.install_pca(q.meta.get("pca_vis"))
            tgt = t_enc.encode_target(scene, det, obj, q.meta.get("canon_frame"))
            if len(tgt.pts) >= 4 and cache:
                cache.put_arrays("target", key, pts=tgt.pts, feats=tgt.feats)
            tgt.meta["feats_w1"] = tgt.feats
        if args.icp_dense and len(tgt.pts) >= 4:
            # Rebuilt here rather than cached: it is a pure depth back-projection
            # (no network), so it costs microseconds, and keeping it OUT of the
            # cached arrays is what lets an existing feature cache still hit.
            tgt.pts_dense = dense_mask_cloud(scene, det.mask, args.icp_dense_max)
            dense_sizes.append(0 if tgt.pts_dense is None else len(tgt.pts_dense))
        return tgt

    n_done = 0
    # Stage failures are counted and NEVER silent: the old bare `except
    # Exception: continue` turned real bugs (stale signatures, dim mismatches)
    # into rows of zeros that looked like "object not found". The run still
    # continues — a 20h benchmark should survive one bad image — but every
    # failure prints, the first of each (stage, type) prints its traceback,
    # and the run ends with a summary instead of a clean-looking "done".
    failures: dict = {}

    def note_failure(stage, label, e):
        import traceback
        key = (stage, type(e).__name__)
        failures[key] = failures.get(key, 0) + 1
        print(f"[FAIL {stage}] {label}: {type(e).__name__}: {e}", flush=True)
        if failures[key] == 1:
            traceback.print_exc()

    header_needed = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        wr = csv.writer(f)
        if header_needed:
            wr.writerow(["scene_id", "im_id", "obj_id", "score", "R", "t", "time"])
        for (scene_id, im_id), obj_ids in sorted(by_img.items()):
            pending = [(o, n) for o, n in obj_ids
                       if (scene_id, im_id, o) not in done]
            if not pending:
                continue
            sdir = bop / layout["split"] / f"{scene_id:06d}"
            try:
                # scene_camera.json inside the try: a missing SCENE (wrong
                # --split, partial download) must hit the same fatal-or-skip
                # policy as a missing image, not leak a bare traceback.
                cams = json.load(open(sdir / "scene_camera.json"))
                if str(im_id) not in cams:
                    raise SystemExit(
                        f"im_id {im_id} absent from {sdir}/scene_camera.json "
                        f"— the targets file and the data tree disagree; not "
                        f"skippable.")
                cam = cams[str(im_id)]
                K = np.array(cam["cam_K"]).reshape(3, 3)
                rgb_bgr, depth_raw = read_frame_images(sdir, im_id, layout)
            except FileNotFoundError as e:
                # An unreadable image is FATAL by default. Writing zero rows
                # here (the old behaviour) has the property that a wrong
                # layout — every image "missing" — completes the entire
                # dataset as a clean-looking all-zero CSV instead of crashing.
                if not args.allow_missing_images:
                    raise SystemExit(
                        f"cannot read {e} (file missing or undecodable).\n"
                        f"Aborting rather than writing zero rows "
                        f"(--allow-missing-images opts back in). If EVERY "
                        f"image fails, the layout is wrong — check the "
                        f"dataset/split/models line printed at startup; a "
                        f"single failure is more likely a corrupt file.")
                # Opt-in skip keeps the completion invariant: inst_count rows
                # per target, or resume classifies this as a mid-target crash
                # and re-runs it forever (the image will still be missing).
                # Counted, so the run cannot end with a clean-looking "done".
                note_failure("load_image", f"im {scene_id}/{im_id}", e)
                for o, n in pending:
                    for _ in range(n):
                        wr.writerow([scene_id, im_id, o, 0.0, IDN, ZT, "0.0"])
                continue
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            depth = depth_raw.astype(np.float32) * cam["depth_scale"] / 1000.0
            scene = Scene(rgb=rgb, depth=depth, K=K,
                          scene_id=scene_id, im_id=im_id)
            scene_fp = fingerprint(rgb, depth, K) if cache else None

            # Buffer per-object hyp maps so --dual-assign can re-pick after
            # both confusable CADs have registered the same cand indices.
            buffered: dict = {}  # obj_id -> (inst_count, hyps_by_det, elapsed)

            for obj_id, inst_count in pending:
                t_start = time.time()
                obj, q, (solver, refiner, scorer) = query_cache[obj_id]
                frame = q.meta.get("canon_frame")
                # Hypotheses grouped per detection: one detection ≈ one
                # candidate instance, and inst_count rows come from distinct
                # detections (see adapters.select_top_instances).
                hyps_by_det: dict = {}
                try:
                    dets = segmentor.segment(scene, obj)
                except Exception as e:
                    note_failure("segment", f"obj{obj_id}", e)
                    dets = []
                for ci, det in enumerate(dets):
                    try:
                        tgt = encode_target_cached(scene, det, obj, q, ci,
                                                   scene_fp)
                    except Exception as e:
                        # Often a degenerate mask cloud (e.g. GeDi LRF) — but
                        # logged, because "often" is not "always".
                        note_failure("encode_target", f"obj{obj_id}", e)
                        continue
                    if len(tgt.pts) < 4:
                        continue
                    if probe_wr is not None:
                        if scene_id not in probe_gt_cache:
                            probe_gt_cache[scene_id] = json.load(
                                open(sdir / "scene_gt.json"))
                        gts = [dict(R=np.array(g["cam_R_m2c"]).reshape(3, 3),
                                    t=np.array(g["cam_t_m2c"], dtype=float))
                               for g in probe_gt_cache[scene_id].get(str(im_id), [])
                               if g["obj_id"] == obj_id]
                        if gts:
                            r1, r10, reach, med1, tau_mm = probe_corr_stats(
                                q, tgt, gts, probe_syms[obj_id],
                                diameters_m[obj_id])
                        else:
                            # A detection with no GT instance (false-positive
                            # mask): recorded, not scored - dropping it would
                            # overstate the pool quality.
                            r1 = r10 = reach = med1 = tau_mm = -1.0
                        probe_wr.writerow([scene_id, im_id, obj_id, ci,
                                           len(tgt.pts), len(q.pts), len(gts),
                                           f"{r1:.4f}", f"{r10:.4f}",
                                           f"{reach:.4f}", f"{med1:.2f}",
                                           f"{tau_mm:.2f}"])
                        continue
                    for w in weights:
                        qw = q if w == 1.0 else _reweighted(q, w, vis_split)
                        tw = tgt if w == 1.0 else _reweighted(tgt, w, vis_split)
                        try:
                            for h in solver.solve(qw, tw, frame):
                                h = refiner.refine(h, scene, obj, qw, tw)
                                h = scorer.score(h, qw, tw)
                                hyps_by_det.setdefault(ci, []).append(h)
                                if cand_f is not None:
                                    cand_wr.writerow([
                                        scene_id, im_id, obj_id, ci, w,
                                        f"{h.breakdown.get('s_icp', 0):.4f}",
                                        f"{h.breakdown.get('s_feat_1', 0):.4f}",
                                        f"{h.breakdown.get('metric_fit', 1):.4f}",
                                        f"{h.score:.6f}",
                                        " ".join(f"{v:.6f}" for v in h.R.flatten()),
                                        " ".join(f"{v:.4f}" for v in (h.t * 1000.0))]
                                        + ([f"{h.breakdown['s_coarse']:.4f}",
                                            " ".join(f"{v:.6f}" for v in
                                                     h.breakdown["R_coarse"].flatten()),
                                            " ".join(f"{v:.4f}" for v in
                                                     (h.breakdown["t_coarse"] * 1000.0))]
                                           if args.score_coarse else [])
                                        + [args.solver])
                        except Exception as e:
                            note_failure("solve/refine/score", f"obj{obj_id}", e)
                            continue
                elapsed = f"{time.time()-t_start:.3f}"
                buffered[obj_id] = (inst_count, hyps_by_det, elapsed)

            if probe_wr is not None:
                # Probe mode never writes pose rows: the out CSV stays empty
                # and resume semantics do not apply. The probe CSV is the
                # product.
                n_done += len(pending)
                continue

            for obj_id, (inst_count, hyps_by_det, elapsed) in buffered.items():
                # THE COMPLETION INVARIANT: a finished target emits EXACTLY
                # inst_count rows — champions first, zero rows (score 0,
                # identity R) padding the rest. Resume can then classify by
                # row COUNT alone. inst_count==1: one champion or one zero.
                if args.dual_assign and inst_count == 1:
                    pid = partner_id(obj_id, merge)
                    if pid is not None and pid in buffered:
                        best = dual_assign_hyps(
                            hyps_by_det, buffered[pid][1], use_metric_fit=True
                        )
                        champs = [best] if best is not None else []
                    else:
                        champs = select_top_instances(
                            hyps_by_det, selector, inst_count)
                else:
                    champs = select_top_instances(
                        hyps_by_det, selector, inst_count)
                for best in champs:
                    wr.writerow([scene_id, im_id, obj_id,
                                 f"{best.score:.6f}",
                                 " ".join(f"{v:.6f}" for v in best.R.flatten()),
                                 " ".join(f"{v:.4f}" for v in (best.t * 1000.0)),
                                 elapsed])
                for _ in range(inst_count - len(champs)):
                    wr.writerow([scene_id, im_id, obj_id, 0.0, IDN, ZT,
                                 elapsed])
                f.flush()
                n_done += 1
                if n_done % 50 == 0:
                    print(f"{n_done} targets this run", flush=True)
    if cand_f is not None:
        cand_f.close()
    if probe_wr is not None:
        probe_f.close()
        rows_p = list(csv.DictReader(open(args.probe_corr)))
        scored = [r for r in rows_p if float(r["n_gt"]) > 0]
        if scored:
            a = np.array([[float(r["rate1"]), float(r["rate10"]),
                           float(r["reach10"])] for r in scored])
            print(f"probe: {len(rows_p)} (target,mask) rows, {len(scored)} "
                  f"with GT | mean rate1={a[:,0].mean():.4f} "
                  f"rate10={a[:,1].mean():.4f} reach10={a[:,2].mean():.4f} | "
                  f"median rate1={np.median(a[:,0]):.4f}", flush=True)
        print(f"probe -> {args.probe_corr}", flush=True)
    if dense_sizes:
        # Positive evidence that --icp-dense actually reached ICP: an empty or
        # sparse-looking distribution here means the fix did nothing.
        s = np.array(dense_sizes)
        print(f"icp-dense: {len(s)} target clouds, |P_T^dense| "
              f"min/median/max = {s.min()}/{int(np.median(s))}/{s.max()} "
              f"(cap {args.icp_dense_max or 'none'})", flush=True)
    if failures:
        total = sum(failures.values())
        print(f"done WITH {total} stage failure(s) -> {args.out}", flush=True)
        for (stage, etype), n in sorted(failures.items()):
            print(f"  {stage}: {etype} x{n}", flush=True)
        print("  affected targets wrote zero/degraded rows and are marked done;"
              " delete their rows from the CSV to re-run them.", flush=True)
    else:
        print(f"done -> {args.out}", flush=True)


def _reweighted(pf, w: float, vis_dim: int | None = None):
    # vis_dim is passed explicitly rather than inferred: the visual part is
    # `vis_dim` wide, not necessarily half (see recipes.scale_vis).
    return PointFeatures(pts=pf.pts,
                         feats=scale_vis(pf.meta["feats_w1"], w, vis_dim),
                         pts_dense=pf.pts_dense, meta=pf.meta)


if __name__ == "__main__":
    main()
