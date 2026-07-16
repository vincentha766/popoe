"""No-GPU end-to-end smoke for the file-based detection-union path.

Exercises the whole non-GPU half of the segmentation stage on REAL data:

    detections load (field coercion) -> RLE mask decode -> N-way top-M union
    with per-source provenance -> instance selection (adapters.select_top_instances)

and prints a short union-shape / source-distribution report to stdout.

Why this is GPU-free by construction: `BOPDetectionsSegmentor.segment()` reads
only `scene.scene_id` / `scene.im_id` to look up candidates — the masks come
from the stored RLE, never from the image — so we drive it with placeholder
Scenes and never touch RGB-D or a CAD model. Instance selection is exercised on
trivial `PoseHypothesis`es whose score IS the detector score: that plumbing
(union -> per-detection grouping -> top-`inst_count` champions) is what we are
smoke-testing, NOT pose accuracy, which needs the DINOv2/GeDi/RANSAC stack.

`inst_count` is taken as 1 (every YCB-V / LM-O target is single-instance), so
selection reduces to one champion per (image, object) — the historical row.

Usage:
    python examples/union_smoke.py --dataset ycbv          # CNOS+NIDS defaults
    python examples/union_smoke.py --dataset ycbv \
        --source sam6d=/path/sam6d_ycbv.json               # add a third source
    python examples/union_smoke.py --dataset lmo \
        --source cnos=/other/cnos_lmo.json                 # repoint a default
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

import numpy as np

from popoe.adapters import BestScoreSelector, select_top_instances
from popoe.interfaces import ObjectModel, PoseHypothesis, Scene
from popoe.segmentor_detections import (DetectionSource,
                                        BOPDetectionsSegmentor, _coerce_sources)
from popoe.recipes import YCBV_MERGE_LABELS

# Repo-relative default detection files (downloaded under data/, gitignored —
# see README "Detections"). CNOS + NIDS give a real two-way union out of the box.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_sources(dataset: str) -> list[str]:
    det = os.path.join(_ROOT, "data", "detections")
    return [
        f"cnos={os.path.join(det, 'cnos', f'cnos-fastsam_{dataset}-test.json')}",
        f"nids={os.path.join(det, 'nids', f'nids_wa_sappe_{dataset}.json')}",
    ]


def _placeholder_scene(scene_id: int, im_id: int) -> Scene:
    # segment() never reads these arrays; only the ids matter.
    z = np.zeros((1, 1), np.float32)
    return Scene(rgb=np.zeros((1, 1, 3), np.uint8), depth=z, K=np.eye(3),
                 scene_id=scene_id, im_id=im_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ycbv", choices=["ycbv", "lmo"])
    ap.add_argument("--source", action="append", default=[],
                    help="name=path; repeatable. Overrides a default of the "
                         "same name (repoint cnos/nids) or adds a new one "
                         "(e.g. sam6d=... once a file exists).")
    ap.add_argument("--images", type=int, default=50,
                    help="how many (scene,image) to probe")
    ap.add_argument("--topk", type=int, default=2, help="per-source top-M")
    ap.add_argument("--merge", default="ycbv", choices=["ycbv", "none"],
                    help="label pooling for the YCB-V clamp pair")
    args = ap.parse_args()
    if args.images <= 0 or args.topk <= 0:
        raise SystemExit("--images and --topk must be positive")

    # Assemble sources: CNOS+NIDS defaults, with --source overriding a default of
    # the SAME name (repoint cnos) or adding a new one (sam6d=...). Skip files
    # that do not exist, loudly.
    by_name: dict = {s.name: s for s in _coerce_sources(_default_sources(args.dataset))}
    for s in _coerce_sources(args.source) if args.source else []:
        by_name[s.name] = s                  # last wins: override or add
    sources: list[DetectionSource] = []
    for s in by_name.values():
        if os.path.exists(s.path):
            sources.append(s)
        else:
            print(f"[skip] source {s.name!r}: file not found -> {s.path}")
    if not sources:
        raise SystemExit("no detection source files exist; nothing to smoke")

    merge = YCBV_MERGE_LABELS if (args.merge == "ycbv"
                                  and args.dataset == "ycbv") else None

    print("=== popoe detection-union smoke (no GPU) ===")
    print(f"dataset: {args.dataset} | topk(per-source)={args.topk} | "
          f"merge={'ycbv-clamp' if merge else 'none'}")
    seg = BOPDetectionsSegmentor(sources=sources, topk=args.topk,
                                 merge_labels=merge)

    # Per-source loaded record counts (from the segmentor's own index).
    loaded = Counter()
    for recs in seg._by_img.values():
        for d in recs:
            loaded[d["source"]] += 1
    print("sources: " + " | ".join(
        f"{s.name}={loaded[s.name]} recs" for s in sources)
        + f" over {len(seg._by_img)} images")
    if not any(s.name == "sam6d" for s in sources):
        print("NOTE: no 'sam6d' source present — SAM-6D ISM emits no committed "
              "detections file locally (it runs on a GPU pod), so the intended "
              "three-way CNOS+SAM-6D+NIDS union is exercised as its available "
              f"{len(sources)}-way subset. Pass --source sam6d=<file> to add it.")

    # Probe the first N images. Probes are DETECTION-LABEL derived (there is no
    # BOP test_targets file in this no-dataset smoke): each image is queried for
    # every object whose label appears in it, PLUS merge partners — so obj20 is
    # probed when only label-19 masks are present and the pooling path runs.
    images = sorted(seg._by_img)[: args.images]
    selector = BestScoreSelector()
    n_probes = n_zero = 0
    cand_sources = Counter()
    champ_sources = Counter()
    shapes = set()
    dtypes = set()
    min_area = None
    for (scene_id, im_id) in images:
        scene = _placeholder_scene(scene_id, im_id)
        present = {d["category_id"] for d in seg._by_img[(scene_id, im_id)]}
        probe = set(present)
        for oid, pooled in (merge or {}).items():
            if set(pooled) & present:
                probe.add(oid)               # e.g. obj20 pooled from label 19
        for oid in sorted(probe):
            n_probes += 1
            dets = seg.segment(scene, ObjectModel(obj_id=oid, mesh_path="",
                                                  diameter=0.1))
            if not dets:
                n_zero += 1                  # label present but dropped by dedup/min_pixels
                continue
            for d in dets:
                cand_sources[d.source] += 1
                shapes.add(d.mask.shape)
                dtypes.add(str(d.mask.dtype))
                a = int(d.mask.sum())
                min_area = a if min_area is None else min(min_area, a)
            # instance-selection PLUMBING on raw-detector-score hypotheses
            # (one champion, k=inst_count=1). NB: raw scores are NOT comparable
            # across sources (a DINO cosine vs a SAM IoU), so the champion
            # source share below is a plumbing artefact, not the real ranking —
            # that needs ChampionScorer on GPU features.
            hyps_by_det = {
                ci: [PoseHypothesis(R=np.eye(3), t=np.zeros(3), score=d.score,
                                    breakdown={"source": d.source})]
                for ci, d in enumerate(dets)}
            for c in select_top_instances(hyps_by_det, selector, k=1):
                champ_sources[c.breakdown["source"]] += 1

    total_cands = sum(cand_sources.values())
    if total_cands == 0:
        raise SystemExit("union produced ZERO candidates over the probed "
                         "images — the smoke did not exercise decode/selection")
    print(f"\n--- union over {n_probes} detection-label probes "
          f"[{len(images)} images] ---")
    print(f"probes with >=1 candidate: {n_probes - n_zero} | "
          f"zero after dedup/min_pixels: {n_zero}")
    print(f"candidates: total {total_cands} | "
          f"avg/probe {total_cands / n_probes:.2f}")
    print("candidate source distribution: " + ", ".join(
        f"{k} {v} ({v / total_cands:.0%})"
        for k, v in cand_sources.most_common()))
    print(f"mask decode: shapes={sorted(shapes)} dtypes={sorted(dtypes)} "
          f"min_area={min_area} (>= min_pixels={seg.min_pixels})")
    tot_ch = sum(champ_sources.values())
    print(f"instance-selection plumbing: {tot_ch} champions (k=inst_count=1) | "
          "raw-score source share (NOT the real ranking): " + ", ".join(
              f"{k} {v / tot_ch:.0%}" for k, v in champ_sources.most_common()))
    print("\nOK: load -> decode -> N-way union -> selection ran end-to-end.")


if __name__ == "__main__":
    main()
