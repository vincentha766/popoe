"""Replay arbitration rules over a --cand-csv dump — csv module, zero GPU.

See the module docstring history: product rules over dumped score columns,
one champion per (scene, im, obj).
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict

KEY = ["scene_id", "im_id", "obj_id"]
RESULT = KEY + ["score", "R", "t"]
ZERO_R = "1 0 0 0 1 0 0 0 1"
ZERO_T = "0 0 0"
CAND = ["cand", "w"]
NON_TERMS = (set(KEY) | set(CAND)
             | {"R", "t", "time", "solver", "source", "R_coarse", "t_coarse",
                "R_prererank", "t_prererank"})


def _key(row):
    return int(row["scene_id"]), int(row["im_id"]), int(row["obj_id"])


def parse_rule(rule: str, columns) -> dict:
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
            raise SystemExit(
                f"rule {rule!r}: negative exponent on {name!r} is not allowed "
                f"(clamped evidence would become +inf)")
        terms[name] = terms.get(name, 0.0) + e
    if not terms:
        raise SystemExit(f"rule {rule!r} has no terms")
    return terms


def _row_score(row: dict, terms: dict) -> float:
    s = 1.0
    for name, exp in terms.items():
        s *= max(float(row[name]), 0.0) ** exp
    return s


def rule_score(rows: list, terms: dict) -> list:
    return [_row_score(r, terms) for r in rows]


def champion_index(rows: list, scores) -> dict:
    """KEY tuple -> row index maximising scores[i]."""
    best = {}
    for i, row in enumerate(rows):
        k = _key(row)
        if k not in best or float(scores[i]) > float(scores[best[k]]):
            best[k] = i
    return best


def champions(rows: list, scores) -> list:
    idx = champion_index(rows, scores)
    return [rows[i] for i in idx.values()]


def load_target_universe(path: str) -> list:
    with open(path, newline="") as f:
        targets = list(csv.DictReader(f))
    missing = [c for c in KEY if c not in (targets[0] if targets else {})]
    if missing:
        raise SystemExit(f"{path} missing target key columns {missing}")
    seen = set()
    for r in targets:
        k = _key(r)
        if k in seen:
            raise SystemExit(f"{path} has duplicate target keys, e.g. {dict(zip(KEY, k))}")
        seen.add(k)
    return [{c: r[c] for c in KEY} for r in targets]


def complete_results(champs: list, targets: list) -> tuple:
    by_key = {}
    for r in champs:
        k = _key(r)
        if k in by_key:
            raise SystemExit(f"replay has duplicate target keys, e.g. {dict(zip(KEY, k))}")
        by_key[k] = {c: r[c] for c in RESULT}
    known = {_key(t) for t in targets}
    extra = [k for k in by_key if k not in known]
    if extra:
        raise SystemExit(
            f"cand dump contains {len(extra)} targets outside --target-csv, "
            f"e.g. {extra[:3]}")
    out, n_missed = [], 0
    for t in targets:
        k = _key(t)
        if k in by_key:
            out.append(by_key[k])
        else:
            n_missed += 1
            out.append({**{c: t[c] for c in KEY},
                        "score": 0.0, "R": ZERO_R, "t": ZERO_T})
    return out, n_missed


def _split_values(values) -> list:
    out: list = []
    for chunk in values or []:
        out.extend(v.strip() for v in chunk.split(",") if v.strip())
    return out


def restrict(rows: list, column: str, wanted: list, cast=lambda v: v):
    if not rows or column not in rows[0]:
        raise SystemExit(
            f"dump has no {column!r} column — cannot restrict by it "
            f"(columns: {sorted(rows[0]) if rows else []})")
    present = [cast(r[column]) for r in rows]
    available = sorted(set(p for p in present if p is not None))
    keep = [cast(v) for v in wanted]
    unknown = [v for v in keep if v not in set(available)]
    if unknown:
        raise SystemExit(
            f"--keep-{column} value(s) {unknown} absent from the dump "
            f"(present: {available})")
    kept = [r for r, p in zip(rows, present) if p in set(keep)]
    return kept, f"{column}={'+'.join(str(v) for v in keep)}"


def _round6(v):
    try:
        return round(float(v), 6)
    except (TypeError, ValueError):
        return v


def _slug(rule: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", rule).strip("_")


def _n_targets(rows):
    return len({_key(r) for r in rows})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cand_csv")
    ap.add_argument("--rule", action="append", required=True)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--target-csv", default="")
    ap.add_argument("--baseline-col", default="score")
    ap.add_argument("--keep-w", action="append", default=[])
    ap.add_argument("--keep-obj", action="append", default=[])
    ap.add_argument("--keep-source", action="append", default=[])
    args = ap.parse_args()

    with open(args.cand_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{args.cand_csv} has no rows")
    for c in KEY + CAND:
        if c not in rows[0]:
            raise SystemExit(f"{args.cand_csv} missing required column {c!r}")
    targets = load_target_universe(args.target_csv) if args.target_csv else None

    n_all, arm = len(rows), []
    targets_all = _n_targets(rows)
    for col, wanted, cast in (("w", _split_values(args.keep_w), _round6),
                              ("obj_id", _split_values(args.keep_obj), int),
                              ("source", _split_values(args.keep_source), str)):
        if wanted:
            rows, label = restrict(rows, col, wanted, cast)
            arm.append(label)
    if arm:
        if not rows:
            raise SystemExit(f"filter {' '.join(arm)} kept 0 hypotheses")
        kept_targets = _n_targets(rows)
        print(f"filter {' '.join(arm)}: kept {len(rows)}/{n_all} hypotheses, "
              f"{kept_targets}/{targets_all} candidate-bearing targets "
              f"({targets_all - kept_targets} lost every candidate)")
        if targets is None and kept_targets < targets_all:
            print("  WARNING: restriction emptied some targets; without "
                  "--target-csv they vanish from the denominator and the AR is "
                  "NOT comparable to the unrestricted arm")
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    if rows and "solver" in rows[0]:
        solvers = sorted({r["solver"] for r in rows if r.get("solver")})
        if len(solvers) > 1:
            print(f"WARNING: dump mixes {len(solvers)} solvers {solvers} — "
                  "champions are selected ACROSS solvers (a target may pick a "
                  "different solver than its neighbour). Filter the CSV to ONE "
                  "solver for a clean per-solver comparison; the B layer is "
                  "reported as an independent solver configuration.")

    base_idx = champion_index(rows, [float(r[args.baseline_col]) for r in rows])
    n_targets = len(base_idx)
    print(f"{args.cand_csv}: {len(rows)} hypotheses, {n_targets} targets "
          f"(baseline = {args.baseline_col!r})")
    if targets is None:
        print("NOTE: only candidate-bearing targets are here — targets a detector "
              "missed entirely have no cand rows and are ABSENT. AR over the "
              "output CSV is a ceiling; pass --target-csv to zero-pad them. The "
              "flip counts below are exact (same candidate set).")
    else:
        print(f"target universe: {len(targets)} targets from {args.target_csv}")

    for rule in args.rule:
        terms = parse_rule(rule, rows[0].keys())
        scores = rule_score(rows, terms)
        r_idx = champion_index(rows, scores)
        flips = sum(1 for k, i in r_idx.items() if base_idx.get(k) != i)
        print(f"  rule {rule!r}: {flips}/{n_targets} targets flip vs baseline "
              f"({flips / n_targets:.1%})")
        if args.out_dir:
            stem = "_".join([_slug(rule)] + [_slug(a) for a in arm])
            out = os.path.join(args.out_dir, f"replay_{stem}.csv")
            champs = []
            for k, i in r_idx.items():
                row = dict(rows[i])
                row["score"] = scores[i]
                champs.append(row)
            result = champs
            if targets is not None:
                result, n_missed = complete_results(result, targets)
                print(f"    complete denominator: {len(result)} targets "
                      f"({n_missed} zero-padded)")
            with open(out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=RESULT)
                w.writeheader()
                for r in result:
                    w.writerow({c: r[c] for c in RESULT})
            print(f"    -> {out}")


if __name__ == "__main__":
    main()
