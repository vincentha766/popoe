"""Row-by-row A/B between two pose CSVs: what a change FIXED, and what it BROKE.

An AR delta alone cannot tell "lifted 40 targets" from "lifted 90 and wrecked
50". This scores both CSVs per row and reports both directions per leg.

Two readings of "correct" are printed, because neither alone is honest:

  * AR contribution — each row's own contribution to AR, i.e. the fraction of
    the 10 BOP thresholds it passes. Threshold-free, and its mean IS the
    (flat) AR, so improved/broken here account for the headline delta exactly.
  * Strict operating point — MSSD < 0.1 x diameter, MSPD < 20 px, VSD < 0.3.
    A blunt pass/fail that matches how a downstream grasp would see it.

VSD is not re-rendered: popoe's vsd.py persists per-row per-tau errors next to
each CSV as `<csv>.vsd_errs.npz`, keyed by object with rows in CSV order. That
alignment is only valid when both runs emitted the SAME (scene, im, obj)
sequence, which is checked here and fatal when it fails.

Usage:
  python ab_delta.py <base.csv> <new.csv> <bop_root> [label]
"""
from __future__ import annotations

import csv
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

_AR_FLAT = Path(__file__).resolve().parent / "ar_flat.py"
_spec = importlib.util.spec_from_file_location("ar_flat", _AR_FLAT)
ar_flat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ar_flat)

MSSD_THRS = ar_flat.MSSD_THRS
MSPD_THRS = ar_flat.MSPD_THRS
VSD_THS = ar_flat.VSD_THS


def keyed_errors(csv_path, bop_root: Path):
    """{(scene, im, obj): (mssd, mspd, diameter)} plus the row key ORDER."""
    rows = list(csv.DictReader(open(csv_path)))
    order = [(int(r["scene_id"]), int(r["im_id"]), int(r["obj_id"]))
             for r in rows]
    errs = ar_flat.errors(csv_path, bop_root)
    # ar_flat.errors drops rows with no GT and returns (obj, mssd, mspd, diam)
    # in row order, so re-key it against the surviving keys in the same order.
    gt_keys = [k for k in order]
    out = {}
    if len(errs) == len(order):
        for k, e in zip(gt_keys, errs):
            out[k] = (e[1], e[2], e[3])
    else:                      # some rows had no GT — re-derive which survived
        it = iter(errs)
        for k in order:
            e = next(it, None)
            if e is None:
                break
            if e[0] == k[2]:
                out[k] = (e[1], e[2], e[3])
    return out, order


def vsd_rows(csv_path, order):
    """Per-row VSD error vectors aligned to `order`, or None if no npz."""
    path = str(csv_path) + ".vsd_errs.npz"
    if not os.path.exists(path):
        return None
    z = np.load(path)
    cursor = {}
    out = {}
    for k in order:
        oid = k[2]
        arr = z.get(f"obj{oid}") if hasattr(z, "get") else z[f"obj{oid}"]
        i = cursor.get(oid, 0)
        cursor[oid] = i + 1
        out[k] = np.asarray(arr)[i]
    return out


def contrib_mssd(e):
    return float(np.mean([(e[0] < th * e[2]) for th in MSSD_THRS]))


def contrib_mspd(e):
    return float(np.mean([(e[1] < th) for th in MSPD_THRS]))


def contrib_vsd(v):
    return float(np.mean([[(v[i] < th) for th in VSD_THS]
                          for i in range(len(v))]))


def report(name, base, new, strict_base, strict_new):
    b = np.array(base)
    n = np.array(new)
    improved = int((n > b).sum())
    broken = int((n < b).sum())
    sb = np.array(strict_base, bool)
    sn = np.array(strict_new, bool)
    print(f"{name:8}{b.mean():9.4f}{n.mean():9.4f}{(n - b).mean() * 100:+8.2f}"
          f"{improved:9d}{broken:8d}"
          f"{int((~sb & sn).sum()):11d}{int((sb & ~sn).sum()):8d}"
          f"{int(sb.sum()):9d}{int(sn.sum()):7d}")


def main():
    base_csv, new_csv, bop_root = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    label = sys.argv[4] if len(sys.argv) > 4 else f"{Path(new_csv).name} vs base"

    eb, order_b = keyed_errors(base_csv, bop_root)
    en, order_n = keyed_errors(new_csv, bop_root)
    if order_b != order_n:
        raise SystemExit("the two CSVs do not emit the same (scene, im, obj) "
                         "sequence — rows cannot be paired")
    keys = [k for k in order_b if k in eb and k in en]

    vb = vsd_rows(base_csv, order_b)
    vn = vsd_rows(new_csv, order_n)

    print(f"=== {label}  ({len(keys)} paired rows) ===")
    print(f"{'leg':8}{'base':>9}{'new':>9}{'d(pt)':>8}{'improved':>9}"
          f"{'broken':>8}{'strict+':>11}{'strict-':>8}{'sb':>9}{'sn':>7}")
    report("MSSD",
           [contrib_mssd(eb[k]) for k in keys],
           [contrib_mssd(en[k]) for k in keys],
           [eb[k][0] < 0.1 * eb[k][2] for k in keys],
           [en[k][0] < 0.1 * en[k][2] for k in keys])
    report("MSPD",
           [contrib_mspd(eb[k]) for k in keys],
           [contrib_mspd(en[k]) for k in keys],
           [eb[k][1] < 20.0 for k in keys],
           [en[k][1] < 20.0 for k in keys])
    if vb is not None and vn is not None:
        vkeys = [k for k in order_b if k in vb and k in vn]
        report("VSD",
               [contrib_vsd(vb[k]) for k in vkeys],
               [contrib_vsd(vn[k]) for k in vkeys],
               [float(np.mean(vb[k])) < 0.3 for k in vkeys],
               [float(np.mean(vn[k])) < 0.3 for k in vkeys])
    else:
        print("VSD     (no .vsd_errs.npz beside one of the CSVs — leg skipped)")
    print("improved/broken: rows whose AR contribution rose / fell.")
    print("strict+/strict-: rows that crossed the strict operating point "
          "(MSSD<0.1d, MSPD<20px, VSD<0.3) in each direction; sb/sn = how many "
          "pass it before/after.")


if __name__ == "__main__":
    main()
