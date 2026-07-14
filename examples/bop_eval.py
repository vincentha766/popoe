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
from popoe.interfaces import ObjectModel, PointFeatures, PoseHypothesis, Scene
from popoe.recipes import (WEIGHTS, YCBV_MERGE_LABELS, best_encoders,
                           best_segmentor, scale_vis, stages_for_object)

IDN = " ".join(f"{v:.6f}" for v in np.eye(3).flatten())
ZT = "0.0 0.0 0.0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bop", required=True)
    ap.add_argument("--detections", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--objs", default="")
    ap.add_argument("--weights", default=",".join(str(w) for w in WEIGHTS))
    ap.add_argument("--topk", type=int, default=2,
                    help="detections kept per (source,label) bucket; floored "
                         "at the dataset's max inst_count so multi-instance "
                         "targets can receive enough champions")
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--cache", default="", help="target-feature cache dir")
    ap.add_argument("--cand-csv", default="", help="dump every hypothesis")
    ap.add_argument("--merge", default="ycbv",
                    help="'ycbv' for the clamp pair, 'none', or '19:20,...'")
    ap.add_argument("--render-backend", default="nvdiffrast",
                    choices=["nvdiffrast", "trimesh", "auto"],
                    help="CAD renderer for query features. Default demands the "
                         "GPU rasteriser (what the reported numbers used) and "
                         "errors without it; 'auto' accepts the ~100x slower CPU "
                         "ray-caster, which yields DIFFERENT features.")
    args = ap.parse_args()

    bop = Path(args.bop)
    weights = [float(w) for w in args.weights.split(",")]
    obj_filter = {int(x) for x in args.objs.split(",") if x.strip()}
    if args.merge == "ycbv":
        merge = YCBV_MERGE_LABELS
    elif args.merge == "none":
        merge = {}
    else:
        merge = {}
        for grp in args.merge.split(","):
            ids = [int(x) for x in grp.split(":")]
            for a in ids:
                merge[a] = ids
    if args.cache:
        os.makedirs(args.cache, exist_ok=True)

    targets = json.load(open(bop / "test_targets_bop19.json"))
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

    # Detection top-K must not cap output below the largest inst_count, or a
    # 4-instance target could never receive 4 champions (codex round-2 minor).
    max_inst = max(target_counts.values(), default=1)
    segmentor = best_segmentor(args.detections,
                               topk=max(args.topk, max_inst),
                               merge_labels=merge)
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
        "n_points": 3000,
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
        "render_backend": q_enc.render_backend,
    }
    cache = StageCache(args.cache) if args.cache else None

    # Encode ALL queries up front (fail fast; PCA snapshots live in meta).
    # Query features + fitted PCA are CACHED alongside target features: the
    # cached targets' visual halves are projected in the query PCA basis, so
    # re-fitting the PCA in a later run silently breaks them (PCA signs are
    # arbitrary per fit — see fusion.py). One basis per object, persisted.
    query_cache: dict = {}
    for obj_id in sorted({o for objs in by_img.values() for o, _n in objs}):
        obj = ObjectModel(obj_id=obj_id,
                          mesh_path=str(bop / "models" / f"obj_{obj_id:06d}.ply"),
                          diameter=0.0)
        t0 = time.time()
        qkey = (fingerprint("query", enc_cfg, file_fingerprint(obj.mesh_path),
                            obj_id) if cache else None)
        hit = cache.get_arrays("query", qkey) if cache else None
        if hit is not None:
            q = PointFeatures(pts=hit["pts"], feats=hit["feats"],
                              meta={"pca_vis": cache.get_pickle("query", qkey)})
            from popoe.interfaces import CanonFrame
            q.meta["canon_frame"] = CanonFrame.from_points(q.pts)
        else:
            q = q_enc.encode_query(obj)
            if cache:
                cache.put_arrays("query", qkey, pts=q.pts, feats=q.feats)
                cache.put_pickle("query", qkey, q.meta.get("pca_vis"))
        q.meta["qkey"] = qkey
        q.meta["feats_w1"] = q.feats   # genuinely w=1: extraction is pinned
        extent = float(np.ptp(q.pts, axis=0).max())
        stages = stages_for_object(extent, size_aware=obj_id in merge)
        query_cache[obj_id] = (obj, q, stages)
        print(f"  obj{obj_id}: extent={extent*1000:.0f}mm "
              f"encode={time.time()-t0:.1f}s", flush=True)

    cand_f = None
    if args.cand_csv:
        new = not os.path.exists(args.cand_csv)
        cand_f = open(args.cand_csv, "a", newline="")
        cand_wr = csv.writer(cand_f)
        if new:
            cand_wr.writerow(["scene_id", "im_id", "obj_id", "cand", "w",
                              "s_icp", "s_feat_1", "metric_fit", "score",
                              "R", "t"])

    def encode_target_cached(scene, det, obj, q, ci, scene_fp):
        # scene_fp content-hashes rgb/depth/K (invariant 2 in cache.py): the
        # BOP ids alone would silently serve stale features if the files under
        # the same ids ever change (corrected dataset, different --bop root).
        key = (fingerprint("target", enc_cfg, scene_fp,
                           obj.obj_id, det.mask, q.meta["qkey"])
               if cache else None)
        hit = cache.get_arrays("target", key) if cache else None
        if hit is not None:
            return PointFeatures(pts=hit["pts"], feats=hit["feats"],
                                 meta={"feats_w1": hit["feats"]})
        t_enc.install_pca(q.meta.get("pca_vis"))
        tgt = t_enc.encode_target(scene, det, obj, q.meta.get("canon_frame"))
        if len(tgt.pts) >= 4 and cache:
            cache.put_arrays("target", key, pts=tgt.pts, feats=tgt.feats)
        tgt.meta["feats_w1"] = tgt.feats
        return tgt

    n_done = 0
    # Stage failures are counted and NEVER silent: the old bare `except
    # Exception: continue` turned real bugs (stale signatures, dim mismatches)
    # into rows of zeros that looked like "object not found". The run still
    # continues — a 20h benchmark should survive one bad image — but every
    # failure prints, the first of each (stage, type) prints its traceback,
    # and the run ends with a summary instead of a clean-looking "done".
    failures: dict = {}

    def note_failure(stage, obj_id, e):
        import traceback
        key = (stage, type(e).__name__)
        failures[key] = failures.get(key, 0) + 1
        print(f"[FAIL {stage}] obj{obj_id}: {type(e).__name__}: {e}", flush=True)
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
            sdir = bop / "test" / f"{scene_id:06d}"
            cam = json.load(open(sdir / "scene_camera.json"))[str(im_id)]
            K = np.array(cam["cam_K"]).reshape(3, 3)
            depth_raw = cv2.imread(str(sdir / "depth" / f"{im_id:06d}.png"),
                                   cv2.IMREAD_UNCHANGED)
            rgb_bgr = cv2.imread(str(sdir / "rgb" / f"{im_id:06d}.png"))
            # None-check BEFORE cvtColor: cv2.cvtColor(None) raises, which made
            # this branch dead code when the RGB file was the missing one.
            if depth_raw is None or rgb_bgr is None:
                # The completion invariant applies here too: inst_count rows
                # per target, or resume classifies this as a mid-target crash
                # and re-runs it forever (the image will still be missing).
                for o, n in pending:
                    for _ in range(n):
                        wr.writerow([scene_id, im_id, o, 0.0, IDN, ZT, "0.0"])
                continue
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            depth = depth_raw.astype(np.float32) * cam["depth_scale"] / 1000.0
            scene = Scene(rgb=rgb, depth=depth, K=K,
                          scene_id=scene_id, im_id=im_id)
            scene_fp = fingerprint(rgb, depth, K) if cache else None

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
                    note_failure("segment", obj_id, e)
                    dets = []
                for ci, det in enumerate(dets):
                    try:
                        tgt = encode_target_cached(scene, det, obj, q, ci,
                                                   scene_fp)
                    except Exception as e:
                        # Often a degenerate mask cloud (e.g. GeDi LRF) — but
                        # logged, because "often" is not "always".
                        note_failure("encode_target", obj_id, e)
                        continue
                    if len(tgt.pts) < 4:
                        continue
                    for w in weights:
                        qw = q if w == 1.0 else _reweighted(q, w)
                        tw = tgt if w == 1.0 else _reweighted(tgt, w)
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
                                        " ".join(f"{v:.4f}" for v in (h.t * 1000.0))])
                        except Exception as e:
                            note_failure("solve/refine/score", obj_id, e)
                            continue
                champs = select_top_instances(hyps_by_det, selector, inst_count)
                # THE COMPLETION INVARIANT: a finished target emits EXACTLY
                # inst_count rows — champions first, zero rows (score 0,
                # identity R) padding the rest. Resume can then classify by
                # row COUNT alone; inferring completion from row contents
                # cannot work (a legitimate 2-champion inst_count=3 target is
                # indistinguishable from a crash after two rows, and a real
                # score can format as "0.000000"). One row per INSTANCE is
                # BOP-standard; metrics/ar.py + vsd.py assume one row per
                # target and guard against multi-instance CSVs.
                # inst_count==1: one champion row or one zero row — identical
                # to the historical format.
                elapsed = f"{time.time()-t_start:.3f}"
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
    if failures:
        total = sum(failures.values())
        print(f"done WITH {total} stage failure(s) -> {args.out}", flush=True)
        for (stage, etype), n in sorted(failures.items()):
            print(f"  {stage}: {etype} x{n}", flush=True)
        print("  affected targets wrote zero/degraded rows and are marked done;"
              " delete their rows from the CSV to re-run them.", flush=True)
    else:
        print(f"done -> {args.out}", flush=True)


def _reweighted(pf, w: float):
    return PointFeatures(pts=pf.pts, feats=scale_vis(pf.meta["feats_w1"], w),
                         pts_dense=pf.pts_dense, meta=pf.meta)


if __name__ == "__main__":
    main()
