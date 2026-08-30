"""Per-instance A/B: what a pose change fixed, and what it BROKE.

A headline AR delta is a net figure. It cannot tell "+2 pt because 60 instances
improved" apart from "+2 pt because 200 improved and 140 regressed", and only
the second reading warrants caution about the change. So this scores both CSVs
instance by instance and reports the two directions separately.

Per-instance score = that instance's own contribution to BOP AR: the fraction
of (metric, threshold) cells it passes, over MSSD's 10 thresholds, MSPD's 10
and — when the `<csv>.vsd_errs.npz` sidecars exist — VSD's 10. It is in [0, 1],
and its FLAT mean over instances is the flat AR, so the per-instance numbers
add up to the headline by construction (BOP weights instances, not objects).

Two breakage counts, because "broke" has two honest readings:
  strict   the instance's score went DOWN at all (it lost a threshold cell)
  flip     it was mostly-right and became mostly-wrong (>= 0.5 -> < 0.5) — the
           ones a reader would call "used to work, now doesn't"

VSD alignment: popoe's vsd.py appends one row per CSV row, grouped by object in
CSV order, and appends a max-error row for targets with no GT — so the block
for object o lines up with that object's rows in CSV order. That is asserted,
not assumed; a length mismatch aborts rather than silently pairing the wrong
instances.

Usage:
  python scripts/pose_ab_instances.py <before.csv> <after.csv> <bop_root> [label]
"""
from __future__ import annotations

import csv as _csv
import os
import sys
from pathlib import Path

import numpy as np

from popoe.metrics.aggregate import MSPD_THRS, MSSD_THRS, VSD_THS

sys.path.insert(0, os.environ.get("POPOE_BOP_TOOLKIT", "/workspace/bop_toolkit"))
from bop_toolkit_lib import misc, pose_error  # noqa: E402
import json
import trimesh


def load_models(bop_root: Path, obj_ids):
    info = json.load(open(bop_root / "models_eval" / "models_info.json"))
    out = {}
    for o in obj_ids:
        m = trimesh.load(bop_root / "models_eval" / f"obj_{o:06d}.ply",
                         force="mesh")
        out[o] = dict(pts=np.array(m.vertices), diameter=info[str(o)]["diameter"],
                      syms=misc.get_symmetry_transformations(info[str(o)], 0.01))
    return out


def load_gt(bop_root: Path, scenes):
    gt = {}
    for s in scenes:
        sdir = bop_root / "test" / f"{s:06d}"
        scene_gt = json.load(open(sdir / "scene_gt.json"))
        cams = json.load(open(sdir / "scene_camera.json"))
        for im_s, gts in scene_gt.items():
            K = np.array(cams[im_s]["cam_K"]).reshape(3, 3)
            for g in gts:
                gt.setdefault((s, int(im_s), g["obj_id"]), []).append(dict(
                    R=np.array(g["cam_R_m2c"]).reshape(3, 3),
                    t=np.array(g["cam_t_m2c"]).reshape(3, 1), K=K))
    return gt


def per_instance(csv_path, bop_root: Path):
    """(keys, per_leg, leg_names) — per_leg is (N, n_legs) in [0, 1]."""
    rows = list(_csv.DictReader(open(csv_path)))
    keys = [(int(r["scene_id"]), int(r["im_id"]), int(r["obj_id"])) for r in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit(f"{csv_path}: duplicate (scene,im,obj) — this compares "
                         f"single-instance targets only")
    models = load_models(bop_root, sorted({k[2] for k in keys}))
    gt = load_gt(bop_root, sorted({k[0] for k in keys}))

    kept, mssd, mspd, diam = [], [], [], []
    row_idx_of_key = {}
    for i, (r, k) in enumerate(zip(rows, keys)):
        if k not in gt:
            continue
        R = np.array([float(x) for x in r["R"].split()]).reshape(3, 3)
        t = np.array([float(x) for x in r["t"].split()]).reshape(3, 1)
        d = models[k[2]]
        mssd.append(min(pose_error.mssd(R, t, g["R"], g["t"], d["pts"], d["syms"])
                        for g in gt[k]))
        mspd.append(min(pose_error.mspd(R, t, g["R"], g["t"], g["K"], d["pts"],
                                        d["syms"]) for g in gt[k]))
        diam.append(d["diameter"])
        row_idx_of_key[k] = i
        kept.append(k)

    mssd, mspd, diam = map(np.asarray, (mssd, mspd, diam))
    # One score per LEG, then legs averaged with equal weight — BOP's AR is
    # mean(AR_MSSD, AR_MSPD, AR_VSD), and VSD has n_taus x 10 cells against the
    # others' 10, so pooling all cells into one mean would silently give VSD
    # three times the weight (and shifted this script's AR by 1.4 pt when it
    # did, which is how the bug was caught).
    per_leg = [np.stack([(mssd < th * diam) for th in MSSD_THRS], 1).mean(1),
               np.stack([(mspd < th) for th in MSPD_THRS], 1).mean(1)]
    legs = ["MSSD", "MSPD"]

    npz = str(csv_path) + ".vsd_errs.npz"
    if os.path.exists(npz):
        z = np.load(npz)
        # Rebuild the per-object CSV order vsd.py wrote in, then index it by key.
        vsd_of_key = {}
        for oid_key in z.files:
            oid = int(oid_key.replace("obj", ""))
            block = np.asarray(z[oid_key])
            obj_rows = [k for k in keys if k[2] == oid]
            if len(obj_rows) != len(block):
                raise SystemExit(
                    f"{npz}: obj{oid} block has {len(block)} rows but the CSV "
                    f"has {len(obj_rows)} — cannot align VSD per instance")
            for k, e in zip(obj_rows, block):
                vsd_of_key[k] = e
        vsd = np.stack([vsd_of_key[k] for k in kept], 0)      # (N, n_taus)
        per_leg.append(np.stack([(vsd[:, i] < th)
                                 for i in range(vsd.shape[1])
                                 for th in VSD_THS], 1).mean(1))
        legs.append("VSD")
    return kept, np.stack(per_leg, 1), legs


def main():
    before_csv, after_csv, bop_root = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    label = sys.argv[4] if len(sys.argv) > 4 else "A/B"
    kb, cb, legs_b = per_instance(before_csv, bop_root)
    ka, ca, legs_a = per_instance(after_csv, bop_root)
    if kb != ka or legs_b != legs_a:
        raise SystemExit(f"the two arms are not comparable: {len(kb)} vs "
                         f"{len(ka)} instances, legs {legs_b} vs {legs_a}")
    sb, sa = cb.mean(1), ca.mean(1)
    d = sa - sb
    improved, broken = d > 1e-12, d < -1e-12
    flip_break = (sb >= 0.5) & (sa < 0.5)
    flip_fix = (sb < 0.5) & (sa >= 0.5)

    print(f"=== {label}: {len(sb)} scored instances, legs={'+'.join(legs_b)} ===")
    print(f"flat AR   before {sb.mean():.4f}   after {sa.mean():.4f}   "
          f"delta {sa.mean()-sb.mean():+.4f} ({100*(sa.mean()-sb.mean()):+.2f} pt)")
    print(f"improved  : {int(improved.sum()):5d}   (total gain {d[improved].sum():+.2f})")
    print(f"BROKEN    : {int(broken.sum()):5d}   (total loss {d[broken].sum():+.2f})")
    print(f"unchanged : {int((~improved & ~broken).sum()):5d}")
    print(f"clear flips: {int(flip_break.sum())} mostly-right -> mostly-wrong, "
          f"{int(flip_fix.sum())} the other way")
    objs = np.array([k[2] for k in kb])
    print(f"{'obj':>5}{'n':>6}{'before':>9}{'after':>9}{'delta':>9}"
          f"{'improved':>10}{'broken':>8}")
    for o in np.unique(objs):
        m = objs == o
        print(f"{o:5d}{int(m.sum()):6d}{sb[m].mean():9.4f}{sa[m].mean():9.4f}"
              f"{sa[m].mean()-sb[m].mean():+9.4f}"
              f"{int(improved[m].sum()):10d}{int(broken[m].sum()):8d}")


if __name__ == "__main__":
    main()
