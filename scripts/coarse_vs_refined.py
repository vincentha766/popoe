"""How many AR points does ICP actually buy? Split one run into two pose CSVs.

FreeZe's Table 3 reports its pipeline with and without refinement (+4.76 pt
mean from ICP alone). We cannot reproduce that row's semantics: our arbitration
score is `s_icp * s_feat_1`, and `s_icp` IS the ICP fitness — delete ICP and the
selection rule has nothing to rank with. What we CAN measure is narrower and
still decisive:

    select the champion exactly as the live pipeline does (full score, ICP
    included), then score that same champion's PRE-ICP pose against ground
    truth.

That answers "how far did ICP move the answer, and was the move worth points",
holding the selection fixed. It is NOT "the pipeline without refinement": the
candidate set and the winner are chosen with ICP evidence in hand. Any
comparison against FreeZe's no-refinement row has to carry that caveat.

Input is a `bop_eval --cand-csv … --score-coarse` dump, which since this change
carries `R_coarse` / `t_coarse` (the pose ICP started from) beside every row's
refined `R` / `t`. Output is two BOP-format CSVs over the SAME target set:

    <prefix>_refined.csv   the champion's post-ICP pose  (must equal --out)
    <prefix>_coarse.csv    the champion's pre-ICP pose

Both are padded from the run's `--out` CSV, so a target the detector missed
contributes an identical zero row to each and AR is computed over the same
denominator. Feeding only the dump's rows would score the two columns over
different target sets and quietly favour whichever had fewer.

The refined CSV is also a self-check: it is reconstructed from the dump by
argmax, so comparing it to `--out` row by row verifies that this script's
champion is the live pipeline's champion. A nonzero mismatch count means the
reconstruction is wrong and the coarse column cannot be trusted either — the
script exits nonzero so the driver stops instead of scoring a wrong column.

A differing pose at EQUAL score is counted separately as a tie, not a mismatch.
The live scorer ranks in full precision while this script ranks on the 6-dp
string, so a tie can land on either row; both are legitimate argmaxes and each
carries its own coarse pose. Failing on those would abort a healthy run.

Single-instance only (`inst_count == 1`, i.e. all of LM-O / YCB-V / TUD-L):
with several instances per target, "the champion" is a set and reconstructing
which dump row produced which output row needs the live selector's ordering.
Duplicate keys in --out are refused rather than guessed.

Usage:
  python scripts/coarse_vs_refined.py \
      --cand-csv  results/icp_ab_lmo_cands.csv \
      --out-csv   results/icp_ab_lmo.csv \
      --prefix    results/icp_ab_lmo
"""

from __future__ import annotations

import argparse
import csv
import sys

KEY = ("scene_id", "im_id", "obj_id")


def read_out_csv(path):
    """The run's --out rows in file order, refusing multi-instance targets."""
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit(f"{path} has no rows")
    keys = [tuple(r[k] for k in KEY) for r in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit(
            f"{path} has duplicate (scene,im,obj) rows (inst_count > 1). This "
            f"splitter reconstructs ONE champion per target by argmax; with "
            f"several champions per target the dump row -> output row mapping "
            f"is the live selector's, not reproducible here.")
    return rows


def champions_by_key(cand_csv):
    """Per target, the dump row with the highest `score`.

    This is exactly `adapters.select_top_instances(..., k=1)`: that takes the
    best hypothesis per detection and then the best of those, which for k=1 is
    the plain argmax over all of the target's rows. Ties are broken by first
    occurrence, matching Python's max()."""
    best: dict = {}
    with open(cand_csv, newline="") as f:
        rd = csv.DictReader(f)
        for col in ("s_coarse", "R_coarse", "t_coarse"):
            if col not in (rd.fieldnames or []):
                raise SystemExit(
                    f"{cand_csv} has no {col!r} column — re-dump with "
                    f"`bop_eval --cand-csv … --score-coarse`, which is what "
                    f"records the pre-ICP pose.")
        for r in rd:
            k = tuple(r[c] for c in KEY)
            s = float(r["score"])
            if k not in best or s > best[k][0]:
                best[k] = (s, r)
    return {k: v[1] for k, v in best.items()}


def split(cand_csv, out_csv, prefix):
    out_rows = read_out_csv(out_csv)
    champs = champions_by_key(cand_csv)

    ref_path, coarse_path = f"{prefix}_refined.csv", f"{prefix}_coarse.csv"
    header = ["scene_id", "im_id", "obj_id", "score", "R", "t", "time"]
    n_pad = n_mismatch = n_tie = 0
    with open(ref_path, "w", newline="") as fr, \
            open(coarse_path, "w", newline="") as fc:
        wr, wc = csv.writer(fr), csv.writer(fc)
        wr.writerow(header)
        wc.writerow(header)
        for row in out_rows:
            k = tuple(row[c] for c in KEY)
            t_col = row.get("time", "")
            ch = champs.get(k)
            if ch is None:
                # No candidate reached the scorer (detector missed the object,
                # or every mask degenerated). --out already holds the zero row;
                # copy it to BOTH sides so the denominators match.
                n_pad += 1
                base = [row[c] for c in KEY] + [row["score"], row["R"],
                                                row["t"], t_col]
                wr.writerow(base)
                wc.writerow(base)
                continue
            # Same serialisation on both sides (bop_eval writes R at 6 dp and
            # t in mm at 4 dp for the refined AND the coarse pose), so this is
            # an exact string comparison, not a tolerance.
            if ch["R"] != row["R"] or ch["t"] != row["t"]:
                # Two different things wear this disguise, and only one is a
                # bug. If the champion we rebuilt carries a DIFFERENT score
                # than the one the live run wrote, we picked the wrong row and
                # the coarse column beside it belongs to the wrong candidate —
                # fatal. If the scores are equal we hit a tie: the live scorer
                # ranked in full precision and we rank on the 6-dp string, so
                # either row is a legitimate argmax and the coarse column is
                # still that candidate's own. Report it, do not fail on it.
                if ch["score"] == row["score"]:
                    n_tie += 1
                else:
                    n_mismatch += 1
            wr.writerow([*k, ch["score"], ch["R"], ch["t"], t_col])
            wc.writerow([*k, ch["score"], ch["R_coarse"], ch["t_coarse"],
                         t_col])

    print(f"targets={len(out_rows)}  with-candidates={len(out_rows) - n_pad}  "
          f"zero-padded={n_pad}")
    print(f"refined-vs-out mismatches: {n_mismatch}"
          + ("  <-- reconstruction FAILED; the coarse column is not "
             "trustworthy either" if n_mismatch else "  (champion "
             "reconstruction verified)"))
    print(f"score ties resolved to the other row: {n_tie}"
          "  (benign: same score, both rows are argmax)")
    print(f"-> {ref_path}\n-> {coarse_path}")
    return n_mismatch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand-csv", required=True,
                    help="bop_eval --cand-csv dump from a --score-coarse run")
    ap.add_argument("--out-csv", required=True,
                    help="the same run's --out CSV (target list + zero rows)")
    ap.add_argument("--prefix", required=True,
                    help="output prefix; writes <prefix>_{refined,coarse}.csv")
    args = ap.parse_args()
    # Nonzero exit, because the caller is icp_ab_run.sh under `set -e`: a
    # silent exit 0 here let a failed reconstruction flow straight into the AR
    # computation and `touch DONE`. Printing a warning nobody reads is how this
    # project produced an all-zero ITODD CSV that looked finished.
    return 1 if split(args.cand_csv, args.out_csv, args.prefix) else 0


if __name__ == "__main__":
    sys.exit(main())
