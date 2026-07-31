"""Smoke gate for --render-rerank: detect *inflated* flip s_icp (the 07-30 bug).

Reads a `--cand-csv` dump, splits candidates by whether the reranker moved the
rotation (>5 deg vs ``R_prererank``), and compares the two ``s_icp`` medians.

**What this gate is for.** The 2026-07-30 defect re-ICP'd only the flipped
variants, and at a 4-10x too loose threshold, so their ``s_icp`` median ran
3.0-4.1x the unflipped group. ChampionScorer multiplies ``s_icp`` into the
score, so the selector preferred a flip for a measurement reason that had
nothing to do with pose quality (LM-O AR(2/3) 0.2489 as shipped).

**One-sided acceptance (2026-07-31).** After PR #29 every variant is re-ICP'd at
the real tau. Wrong flips then get *lower* fitness — that is correct geometry,
not a measurement bug. A bilateral band ``[lo, hi]`` therefore fails healthy
fixed runs (smoke ratio ~0.17x with AR(2/3) 0.816). The gate now fails **only
on flip inflation** (``ratio > hi``). A low ratio prints a WARN but exits 0.

Usage:
  python check_rerank_symmetry.py <cand.csv> [--hi 1.4] [--lo 0.7]

  --hi  fail if flipped/unflipped median ratio exceeds this (inflation)
  --lo  warn (not fail) if ratio is below this (deflation / quality gap)

Exit 0 = pass (no inflation), 1 = fail (inflation), 2 = cannot judge.
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

MOVED_DEG = 5.0     # below this, the reranker kept the champion orientation


def parse_R(s: str) -> np.ndarray:
    return np.array([float(x) for x in s.split()]).reshape(3, 3)


def angle_deg(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(A @ B.T) - 1) / 2, -1, 1))))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fail only if flipped s_icp is inflated vs unflipped "
                    "(measurement-asymmetry bug). Low ratios warn, do not fail.")
    ap.add_argument("cand_csv")
    ap.add_argument("--hi", type=float, default=1.4,
                    help="max acceptable flipped/unflipped s_icp median ratio "
                         "(fail above; default 1.4)")
    ap.add_argument("--lo", type=float, default=0.7,
                    help="warn if ratio is below this (default 0.7); does not fail")
    args = ap.parse_args()

    flipped: list[float] = []
    kept: list[float] = []
    with open(args.cand_csv) as f:
        rd = csv.DictReader(f)
        missing = {"R", "R_prererank", "s_icp"} - set(rd.fieldnames or [])
        if missing:
            print(f"FAIL cannot judge: {args.cand_csv} lacks {sorted(missing)}. "
                  f"The dump must come from a --render-rerank run.")
            return 2
        for row in rd:
            moved = angle_deg(parse_R(row["R"]), parse_R(row["R_prererank"]))
            (flipped if moved > MOVED_DEG else kept).append(float(row["s_icp"]))

    n_f, n_k = len(flipped), len(kept)
    total = n_f + n_k
    if total == 0:
        print(f"FAIL cannot judge: no candidate rows in {args.cand_csv}")
        return 2
    if n_f == 0:
        print(f"CANNOT JUDGE: the reranker moved no candidate in {total} rows. "
              f"Either the smoke subset is too small or rerank is off — this "
              f"gate proves nothing here. Widen the subset and re-run.")
        return 2
    if n_k == 0:
        print(f"CANNOT JUDGE: every one of {total} candidates was flipped; "
              f"no unflipped group to compare against.")
        return 2

    med_f, med_k = float(np.median(flipped)), float(np.median(kept))
    if med_k <= 0:
        print(f"FAIL cannot judge: unflipped s_icp median is {med_k:.4f}; "
              f"a ratio against it is meaningless.")
        return 2
    ratio = med_f / med_k

    print(f"candidates      : {total}  ({n_f} moved >{MOVED_DEG:g} deg, {n_k} kept)")
    print(f"s_icp median    : flipped {med_f:.4f}   unflipped {med_k:.4f}")
    print(f"ratio           : {ratio:.2f}x   "
          f"fail if >{args.hi:g} (inflation); warn if <{args.lo:g}")

    if ratio > args.hi:
        print(f"FAIL — flipped s_icp is inflated ({ratio:.2f}x > {args.hi:g}). "
              f"This is the 2026-07-30 failure shape: ChampionScorer multiplies "
              f"s_icp into the score, so the selector prefers flips for a "
              f"measurement reason. Check that every variant is re-ICP'd at the "
              f"tau ICP recorded in the breakdown (popoe PR #29).")
        return 1

    if ratio < args.lo:
        print(f"PASS with WARN — flipped s_icp is lower ({ratio:.2f}x < {args.lo:g}). "
              f"After PR #29 this is expected when true flips are geometrically "
              f"worse under the same re-ICP tau; it is not the inflation bug. "
              f"Still confirm score sanity (S2) separately.")
        return 0

    print("PASS — no flip inflation; groups within warn band.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
