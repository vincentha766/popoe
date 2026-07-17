"""Replay arbitration rules over a --cand-csv dump — pure pandas, zero GPU.

bop_eval --cand-csv dumps every (mask x weight) hypothesis with its score
breakdown (s_icp, s_feat_1, metric_fit, score, and s_coarse when the run used
--score-coarse). This tool re-selects the per-target champion under ANY product
rule over those columns, WITHOUT re-running the GPU pipeline, and writes a
BOP-format results CSV (scene_id, im_id, obj_id, score, R, t) that the AR metric
can score directly.

Because the 26-rule ablation shows rules do NOT transfer across datasets, the
rule is a parameter (repeatable --rule), never hard-coded — you re-sweep per
dataset. A rule that names a column the dump lacks (e.g. s_coarse on a dump
produced without --score-coarse) is a LOUD error telling you to re-dump with
--score-coarse, never a silent 0.

Rule syntax: a product of terms `name` or `name^exp` (exp >= 0; evidence is
clamped at 0, so a negative exponent — which would send no-evidence to +inf — is
rejected), joined by `*`, over the numeric columns present, e.g.:
    --rule "s_icp * s_feat_1"                 # the champion rule
    --rule "s_icp * s_feat_1 * metric_fit"
    --rule "s_icp * s_feat_1 * s_coarse^0.5"  # needs a --score-coarse dump

Scope: the dump has rows only for candidate-bearing targets, so the replay
covers those; targets a detector missed entirely are absent. Rule-vs-rule flip
counts are exact (same candidate set); AR over the output CSV is a CEILING until
missing targets are zero-padded.

Precision: the dump serialises the term columns to 4 decimals, so replay
recomputes products from ROUNDED values — an approximation of the live scorer's
full-precision argmax. For candidates that tie within ~1e-4 the champion can
differ from a live run; this bounds replay's fidelity (it applies to the
existing champion columns too, not just s_coarse).

Usage:
    python examples/rule_replay.py cands.csv \
        --rule "s_icp*s_feat_1" --rule "s_icp*s_feat_1*s_coarse" --out-dir replays/
"""

from __future__ import annotations

import argparse
import os
import re

import pandas as pd

KEY = ["scene_id", "im_id", "obj_id"]           # one champion per BOP target
# Row identity within a target: which (mask, weight) hypothesis was chosen.
CAND = ["cand", "w"]
# Columns that are NOT numeric rule terms (ids / pose / provenance / solver id).
NON_TERMS = set(KEY) | set(CAND) | {"R", "t", "time", "solver"}


def parse_rule(rule: str, columns) -> dict:
    """Parse 's_icp * s_feat_1^0.5' -> {'s_icp': 1.0, 's_feat_1': 0.5}.

    Every term must be a numeric column present in the dump; a missing one
    (typically s_coarse) raises SystemExit telling you to re-dump, rather than
    being silently treated as 0 or 1."""
    cols = set(columns)
    terms: dict = {}
    for factor in rule.split("*"):
        f = factor.strip()
        if not f:
            continue
        m = re.fullmatch(r"([A-Za-z_]\w*)(?:\^(-?\d+(?:\.\d+)?))?", f)
        if not m:
            raise SystemExit(f"cannot parse rule factor {f!r} in rule {rule!r}")
        name, exp = m.group(1), m.group(2)
        if name in NON_TERMS:
            raise SystemExit(
                f"rule {rule!r}: {name!r} is not a numeric scoring term "
                f"(it is an id / pose / provenance column)")
        if name not in cols:
            hint = (" — re-dump with `bop_eval --cand-csv … --score-coarse`"
                    if name == "s_coarse" else "")
            raise SystemExit(
                f"rule {rule!r} references column {name!r}, not in the dump "
                f"(columns: {sorted(cols - NON_TERMS)}){hint}")
        e = float(exp) if exp else 1.0
        if e < 0:
            # evidence terms are clamped at 0; a negative exponent would turn
            # zero/no-evidence into +inf and let it win. Reject loudly.
            raise SystemExit(
                f"rule {rule!r}: negative exponent on {name!r} is not allowed "
                f"(clamped evidence would become +inf)")
        terms[name] = terms.get(name, 0.0) + e
    if not terms:
        raise SystemExit(f"rule {rule!r} has no terms")
    return terms


def rule_score(df: pd.DataFrame, terms: dict) -> pd.Series:
    """Product of max(col, 0) ** exp — the max(.,0) mirrors the champion rule's
    clamp on s_feat_1 (negative feature scores are non-evidence)."""
    out = pd.Series(1.0, index=df.index)
    for name, exp in terms.items():
        out = out * df[name].clip(lower=0.0) ** exp
    return out


def champion_index(df: pd.DataFrame, score: pd.Series) -> pd.Series:
    """Per (scene, im, obj) target, the DataFrame row label maximising `score`.
    Returned Series is indexed by the KEY tuple, so two rules' choices align by
    target. Row-LABEL identity (not (cand, w)) so a flip is detected even when
    the solver emitted several hypotheses under the same (cand, w) — e.g.
    n_restarts>1 — which share a (cand, w) but differ in pose."""
    return score.groupby([df[k] for k in KEY]).idxmax()


def champions(df: pd.DataFrame, score: pd.Series) -> pd.DataFrame:
    """One row per target: the hypothesis maximising `score`."""
    return df.loc[champion_index(df, score).values]


def _slug(rule: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", rule).strip("_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cand_csv")
    ap.add_argument("--rule", action="append", required=True,
                    help="product rule over dump columns; repeatable")
    ap.add_argument("--out-dir", default="",
                    help="write replay_<rule>.csv results here (optional)")
    ap.add_argument("--baseline-col", default="score",
                    help="column whose champion is the flip baseline (the dump's "
                         "own arbitration score by default)")
    args = ap.parse_args()

    df = pd.read_csv(args.cand_csv)
    if df.empty:
        raise SystemExit(f"{args.cand_csv} has no rows")
    for c in KEY + CAND:
        if c not in df.columns:
            raise SystemExit(f"{args.cand_csv} missing required column {c!r}")
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    if "solver" in df.columns:
        solvers = sorted(df["solver"].dropna().unique())
        if len(solvers) > 1:
            print(f"WARNING: dump mixes {len(solvers)} solvers {solvers} — "
                  "champions are selected ACROSS solvers (a target may pick a "
                  "different solver than its neighbour). Filter the CSV to ONE "
                  "solver for a clean per-solver comparison; the B layer is "
                  "reported as an independent solver configuration.")

    base_idx = champion_index(df, df[args.baseline_col])
    n_targets = len(base_idx)
    print(f"{args.cand_csv}: {len(df)} hypotheses, {n_targets} targets "
          f"(baseline = {args.baseline_col!r})")
    print("NOTE: only candidate-bearing targets are here — targets a detector "
          "missed entirely have no cand rows and are ABSENT. AR over the output "
          "CSV is a ceiling; zero-pad the missing targets to compare against a "
          "full eval run. The flip counts below are exact (same candidate set).")

    for rule in args.rule:
        terms = parse_rule(rule, df.columns)
        r_idx = champion_index(df, rule_score(df, terms))
        # flips vs baseline: targets whose CHOSEN ROW changed (aligned by KEY).
        flips = int((r_idx != base_idx).sum())
        print(f"  rule {rule!r}: {flips}/{n_targets} targets flip vs baseline "
              f"({flips / n_targets:.1%})")
        if args.out_dir:
            out = os.path.join(args.out_dir, f"replay_{_slug(rule)}.csv")
            champs = df.loc[r_idx.values].copy()
            champs["score"] = rule_score(champs, terms).values
            champs[["scene_id", "im_id", "obj_id", "score", "R", "t"]].to_csv(
                out, index=False)
            print(f"    -> {out}")


if __name__ == "__main__":
    main()
