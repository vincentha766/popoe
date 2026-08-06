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
counts are exact (same candidate set). Pass the corresponding full-run
``poses.csv`` as ``--target-csv`` to define the complete target universe and
zero-pad missed targets. Without it, AR over the output CSV is a CEILING.

Precision: the dump serialises the term columns to 4 decimals, so replay
recomputes products from ROUNDED values — an approximation of the live scorer's
full-precision argmax. For candidates that tie within ~1e-4 the champion can
differ from a live run; this bounds replay's fidelity (it applies to the
existing champion columns too, not just s_coarse).

Pool restrictions (`--keep-w`, `--keep-source`) drop candidate rows BEFORE the
argmax, which is a different kind of ablation from a rule change: the rule stays
the frozen one and the question becomes "what could still be selected". A target
whose every candidate is dropped keeps its place in the denominator only via
`--target-csv`, so pass it for any restricted arm you intend to report.

Usage:
    python examples/rule_replay.py cands.csv \
        --target-csv poses.csv \
        --rule "s_icp*s_feat_1" --rule "s_icp*s_feat_1*s_coarse" --out-dir replays/

    # C1 fixed-w arm; C4 source-availability arm
    python examples/rule_replay.py cands.csv --target-csv poses.csv \
        --rule "s_icp*s_feat_1*metric_fit" --keep-w 0.7 --out-dir replays/
    python examples/rule_replay.py cands.csv --target-csv poses.csv \
        --rule "s_icp*s_feat_1*metric_fit" --keep-source cnos,sam6d --out-dir replays/
"""

from __future__ import annotations

import argparse
import os
import re

import pandas as pd

KEY = ["scene_id", "im_id", "obj_id"]           # one champion per BOP target
RESULT = KEY + ["score", "R", "t"]
ZERO_R = "1 0 0 0 1 0 0 0 1"
ZERO_T = "0 0 0"
# Row identity within a target: which (mask, weight) hypothesis was chosen.
CAND = ["cand", "w"]
# Columns that are NOT numeric rule terms (ids / pose / provenance / solver id).
# R_coarse / t_coarse are the pre-ICP POSE (--score-coarse dumps), not evidence:
# without them here, `--rule "…*R_coarse"` would be accepted as a term and
# ranked on a string column.
NON_TERMS = (set(KEY) | set(CAND)
             | {"R", "t", "time", "solver", "source", "R_coarse", "t_coarse",
                "R_prererank", "t_prererank"})


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


def load_target_universe(path: str) -> pd.DataFrame:
    """Load one canonical row per target, preserving the reference order."""
    targets = pd.read_csv(path)
    missing = [c for c in KEY if c not in targets.columns]
    if missing:
        raise SystemExit(f"{path} missing target key columns {missing}")
    targets = targets[KEY].copy()
    duplicated = targets.duplicated(KEY, keep=False)
    if duplicated.any():
        sample = targets.loc[duplicated, KEY].head(3).to_dict("records")
        raise SystemExit(f"{path} has duplicate target keys, e.g. {sample}")
    return targets


def complete_results(champs: pd.DataFrame,
                     targets: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Align champions to the full target universe and zero-pad misses."""
    chosen = champs[RESULT].copy()
    duplicated = chosen.duplicated(KEY, keep=False)
    if duplicated.any():
        sample = chosen.loc[duplicated, KEY].head(3).to_dict("records")
        raise SystemExit(f"replay has duplicate target keys, e.g. {sample}")

    known = pd.MultiIndex.from_frame(targets[KEY])
    observed = pd.MultiIndex.from_frame(chosen[KEY])
    extra = observed.difference(known)
    if len(extra):
        raise SystemExit(
            f"cand dump contains {len(extra)} targets outside --target-csv, "
            f"e.g. {list(extra[:3])}")

    out = targets.merge(chosen, on=KEY, how="left", validate="one_to_one",
                        sort=False)
    missed = out["R"].isna() | out["t"].isna() | out["score"].isna()
    out.loc[missed, "score"] = 0.0
    out.loc[missed, "R"] = ZERO_R
    out.loc[missed, "t"] = ZERO_T
    return out[RESULT], int(missed.sum())


def _split_values(values) -> list:
    """--keep-w 1.0,0.7 --keep-w 0.5 -> ['1.0', '0.7', '0.5']."""
    out: list = []
    for chunk in values or []:
        out.extend(v.strip() for v in chunk.split(",") if v.strip())
    return out


def restrict(df: pd.DataFrame, column: str, wanted: list,
             cast=lambda v: v) -> tuple[pd.DataFrame, str]:
    """Keep only rows whose `column` is in `wanted` — the candidate-availability
    ablations (C1 fixed-w, C4 detection source) are pool restrictions, not rule
    changes, so they must happen BEFORE the champion argmax.

    A value the dump never contains is a LOUD error: silently returning an empty
    or partial arm would read as 'this source buys nothing' when the truth is
    'you misspelled the source'.
    """
    if column not in df.columns:
        raise SystemExit(
            f"dump has no {column!r} column — cannot restrict by it "
            f"(columns: {sorted(df.columns)})")
    present = df[column].map(cast)
    available = sorted(set(present.dropna()))
    keep = [cast(v) for v in wanted]
    unknown = [v for v in keep if v not in set(available)]
    if unknown:
        raise SystemExit(
            f"--keep-{column} value(s) {unknown} absent from the dump "
            f"(present: {available})")
    return df[present.isin(keep)], f"{column}={'+'.join(str(v) for v in keep)}"


def _round6(v):
    try:
        return round(float(v), 6)
    except (TypeError, ValueError):
        return v


def _slug(rule: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", rule).strip("_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cand_csv")
    ap.add_argument("--rule", action="append", required=True,
                    help="product rule over dump columns; repeatable")
    ap.add_argument("--out-dir", default="",
                    help="write replay_<rule>.csv results here (optional)")
    ap.add_argument(
        "--target-csv",
        default="",
        help="full-run poses.csv defining the complete target universe; targets "
             "absent from cand_csv are written as zero-score identity failures")
    ap.add_argument("--baseline-col", default="score",
                    help="column whose champion is the flip baseline (the dump's "
                         "own arbitration score by default)")
    ap.add_argument("--keep-w", action="append", default=[],
                    help="restrict the candidate pool to these visual weights "
                         "(comma-separated or repeatable), e.g. --keep-w 0.7. "
                         "This is the C1 fixed-w arm: adaptive selection is the "
                         "unrestricted run, a fixed w is this pool minus every "
                         "other weight")
    ap.add_argument("--keep-obj", action="append", default=[],
                    help="restrict to these obj_id values (comma-separated or "
                         "repeatable). Use with a --target-csv restricted to the "
                         "same objects — this is the C3 obj19/20 slice, where the "
                         "pooled arms must cover exactly the objects the no-pool "
                         "GPU arm ran")
    ap.add_argument("--keep-source", action="append", default=[],
                    help="restrict the candidate pool to these detection sources "
                         "(comma-separated or repeatable), e.g. "
                         "--keep-source cnos,sam6d. This is the C4 frozen-pool "
                         "source-availability arm; it does NOT re-run detection, "
                         "so it cannot claim end-to-end detector-union equivalence")
    args = ap.parse_args()

    df = pd.read_csv(args.cand_csv)
    if df.empty:
        raise SystemExit(f"{args.cand_csv} has no rows")
    for c in KEY + CAND:
        if c not in df.columns:
            raise SystemExit(f"{args.cand_csv} missing required column {c!r}")
    targets = load_target_universe(args.target_csv) if args.target_csv else None

    n_all, arm = len(df), []
    targets_all = df.groupby(KEY).ngroups
    for col, wanted, cast in (("w", _split_values(args.keep_w), _round6),
                              ("obj_id", _split_values(args.keep_obj), int),
                              ("source", _split_values(args.keep_source), str)):
        if wanted:
            df, label = restrict(df, col, wanted, cast)
            arm.append(label)
    if arm:
        if df.empty:
            raise SystemExit(f"filter {' '.join(arm)} kept 0 hypotheses")
        kept_targets = df.groupby(KEY).ngroups
        print(f"filter {' '.join(arm)}: kept {len(df)}/{n_all} hypotheses, "
              f"{kept_targets}/{targets_all} candidate-bearing targets "
              f"({targets_all - kept_targets} lost every candidate)")
        if targets is None and kept_targets < targets_all:
            print("  WARNING: restriction emptied some targets; without "
                  "--target-csv they vanish from the denominator and the AR is "
                  "NOT comparable to the unrestricted arm")
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
    if targets is None:
        print("NOTE: only candidate-bearing targets are here — targets a detector "
              "missed entirely have no cand rows and are ABSENT. AR over the "
              "output CSV is a ceiling; pass --target-csv to zero-pad them. The "
              "flip counts below are exact (same candidate set).")
    else:
        print(f"target universe: {len(targets)} targets from {args.target_csv}")

    for rule in args.rule:
        terms = parse_rule(rule, df.columns)
        r_idx = champion_index(df, rule_score(df, terms))
        # flips vs baseline: targets whose CHOSEN ROW changed (aligned by KEY).
        flips = int((r_idx != base_idx).sum())
        print(f"  rule {rule!r}: {flips}/{n_targets} targets flip vs baseline "
              f"({flips / n_targets:.1%})")
        if args.out_dir:
            # arm slug in the filename: two restriction arms of the SAME rule
            # would otherwise overwrite each other in one --out-dir.
            stem = "_".join([_slug(rule)] + [_slug(a) for a in arm])
            out = os.path.join(args.out_dir, f"replay_{stem}.csv")
            champs = df.loc[r_idx.values].copy()
            champs["score"] = rule_score(champs, terms).values
            result = champs[RESULT]
            if targets is not None:
                result, n_missed = complete_results(result, targets)
                print(f"    complete denominator: {len(result)} targets "
                      f"({n_missed} zero-padded)")
            result.to_csv(out, index=False)
            print(f"    -> {out}")


if __name__ == "__main__":
    main()
