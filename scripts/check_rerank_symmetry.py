"""Smoke gate for --render-rerank: are flipped and unflipped candidates measured
the same way?

Reads a `--cand-csv` dump, splits candidates by whether the reranker moved the
rotation, and compares the two `s_icp` distributions. They should be close. A
large ratio means the two groups went through different measurements, and since
ChampionScorer multiplies s_icp into the score, the selector will then prefer
one group for a reason that has nothing to do with pose quality.

This is the check that would have caught the 2026-07-30 defect on a one-object
smoke instead of eight full runs: the reranker re-ICP'd only the flips, at a
threshold 4-10x too loose, and their s_icp median ran 3.0-4.1x the unflipped
one (LM-O AR(2/3) 0.2489 as shipped, vs 0.7745 with those picks excluded).

Usage:
  python check_rerank_symmetry.py <cand.csv> [--lo 0.7] [--hi 1.4]

Exit 0 = pass, 1 = fail, 2 = cannot judge (no flips, or column missing).
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
    ap = argparse.ArgumentParser()
    ap.add_argument("cand_csv")
    ap.add_argument("--lo", type=float, default=0.7,
                    help="min acceptable flipped/unflipped s_icp median ratio")
    ap.add_argument("--hi", type=float, default=1.4, help="max acceptable ratio")
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
    print(f"ratio           : {ratio:.2f}x   accept [{args.lo}, {args.hi}]")

    if args.lo <= ratio <= args.hi:
        print("PASS — both groups measured on the same scale.")
        return 0
    print(f"FAIL — flipped and unflipped s_icp are on different scales. "
          f"ChampionScorer multiplies s_icp into the score, so the selector "
          f"will prefer the inflated group regardless of pose quality. "
          f"Check that every variant is re-ICP'd at the tau ICP recorded in "
          f"the breakdown (popoe PR #29).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
