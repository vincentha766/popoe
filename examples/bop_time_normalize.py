"""Normalize a bop_eval pose CSV's `time` column for BOP server submission.

BOP defines `time` as the wall time to estimate poses for ALL objects in the
image, and requires every row of one (scene_id, im_id) to carry the SAME
value (bop_toolkit's check_bop_results enforces it, tolerance 0.001, or a
uniform -1 for "not reported"). bop_eval deliberately writes something else:
the RAW per-target wall time, measured per (image, object) — a resume-safe
measurement worth keeping. This tool derives the compliant column from it:

    time(image) = sum of per-TARGET elapsed over the image's objects
                  [+ the image's detection time from each --detections JSON]

Per TARGET, not per row: a multi-instance target writes inst_count rows that
all carry the target's one elapsed, so summing rows would double-count it.

Detection time is part of the BOP definition for methods consuming
precomputed detections; the official default-detection JSONs record it as a
per-image `time` field on each entry. Pass each detections file the run
consumed (both files of a union — both segmentors ran).

The input is never rewritten in place: the raw measurement is data, the
normalized file is derived from it. `--mode neg1` writes the always-compliant
-1 instead (when timing is not worth reporting, or a legacy CSV has no
usable time column).

Run ONCE, on the raw CSV. The output is not valid input: normalization is a
sum, so summing again multiplies each image's time by its target count — and
the result still passes the server's per-image consistency check, so nothing
downstream would notice. Negative times are refused for the same reason (a
-1 file re-summed would silently submit -N as "not reported").

Usage:
  python examples/bop_time_normalize.py raw.csv --out METHOD_ycbv-test.csv \
      [--detections cnos.json --detections sam6d.json] [--mode sum|neg1]
"""

from __future__ import annotations

import argparse
import csv
import json
import os

HEADER = ["scene_id", "im_id", "obj_id", "score", "R", "t", "time"]


def target_times(rows):
    """Per-target elapsed: {(scene, im, obj): seconds}.

    Rows of one target must agree — bop_eval writes them from a single
    measurement, and resume rewrites a partial target wholly, so disagreement
    means the CSV was not produced by the writer whose invariant we rely on.
    An empty/unparseable value (legacy pre-time-column CSV) is equally fatal;
    both errors name --mode neg1 as the way out.
    """
    out: dict = {}
    for r in rows:
        key = (int(r["scene_id"]), int(r["im_id"]), int(r["obj_id"]))
        try:
            t = float(r.get("time") or "")
        except ValueError:
            raise SystemExit(
                f"row {key} has no usable time value "
                f"({r.get('time', '')!r}); this CSV cannot be normalized by "
                f"summation — use --mode neg1.")
        if t < 0:
            raise SystemExit(
                f"row {key} has negative time {t}. Sum mode needs the RAW "
                f"bop_eval CSV; a -1 (or already-normalized) file cannot be "
                f"re-normalized — re-summing would silently corrupt the "
                f"column while still passing the server's per-image check.")
        if key in out and abs(out[key] - t) > 1e-9:
            raise SystemExit(
                f"target {key} carries two different time values "
                f"({out[key]} vs {t}); not a bop_eval-written CSV — "
                f"use --mode neg1 or fix the input.")
        out[key] = t
    return out


def detection_times(path):
    """Per-image detection seconds from an official-style detections JSON:
    {(scene, im): seconds}, taking max over an image's entries (they carry
    one per-image value; max tolerates rounding drift). A file whose entries
    have no `time` field cannot contribute — fail rather than add zeros."""
    entries = json.load(open(path))
    out: dict = {}
    for e in entries:
        if "time" not in e:
            raise SystemExit(
                f"{path}: detection entries carry no 'time' field, so this "
                f"file cannot contribute detection time. Drop it from "
                f"--detections (accepting pose-only time) or use --mode neg1.")
        key = (int(e["scene_id"]), int(e["image_id"]))
        out[key] = max(out[key], float(e["time"])) if key in out \
            else float(e["time"])
    return out


def normalize(rows, det_maps, mode):
    """Return the rows with the compliant time column; rows keep their order.
    Images absent from a detections file contribute 0 for that file (nothing
    was proposed there) — counted and reported by the caller, not silent."""
    if mode == "neg1":
        return [dict(r, time="-1") for r in rows], 0
    per_target = target_times(rows)
    per_image: dict = {}
    for (s, i, o), t in per_target.items():
        per_image[(s, i)] = per_image.get((s, i), 0.0) + t
    uncovered = 0
    for key in per_image:
        for dmap in det_maps:
            if key in dmap:
                per_image[key] += dmap[key]
            else:
                uncovered += 1
    return [dict(r, time=f"{per_image[(int(r['scene_id']), int(r['im_id']))]:.3f}")
            for r in rows], uncovered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_in", help="raw bop_eval pose CSV")
    ap.add_argument("--out", required=True,
                    help="normalized CSV to write (must differ from the "
                         "input — the raw measurement is not overwritten)")
    ap.add_argument("--detections", action="append", default=[],
                    help="detections JSON whose per-image time the run "
                         "consumed; repeat per source of a union")
    ap.add_argument("--mode", default="sum", choices=["sum", "neg1"],
                    help="sum (default): per-image total of target elapsed "
                         "+ detection time; neg1: uniform -1 (time not "
                         "reported — always compliant)")
    args = ap.parse_args()
    # realpath: "dir/./raw.csv" and symlinks must not slip past a guard whose
    # whole purpose is keeping the raw per-target measurement intact.
    if os.path.realpath(args.out) == os.path.realpath(args.csv_in):
        raise SystemExit("--out must differ from the input; the raw "
                         "per-target measurement is worth keeping.")

    with open(args.csv_in, newline="") as f:
        rd = csv.DictReader(f)
        missing = [c for c in HEADER if c not in (rd.fieldnames or [])]
        if missing:
            raise SystemExit(f"{args.csv_in}: missing column(s) {missing}; "
                             f"expected a bop_eval pose CSV {HEADER}")
        rows = list(rd)

    # neg1 reports no time at all, so the detections files must not even be
    # parsed: a file that would be fatal in sum mode (no time field) cannot
    # block the very mode that error message recommends as the way out.
    if args.mode == "neg1" and args.detections:
        print("--mode neg1 ignores --detections (no time is reported)",
              flush=True)
    det_maps = ([] if args.mode == "neg1"
                else [detection_times(p) for p in args.detections])
    out_rows, uncovered = normalize(rows, det_maps, args.mode)
    if uncovered:
        print(f"{uncovered} (image, detections-file) pairs had no detection "
              f"entry; those images' time is missing that file's detection "
              f"component", flush=True)

    with open(args.out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(HEADER)
        for r in out_rows:
            wr.writerow([r[c] for c in HEADER])
    n_imgs = len({(r["scene_id"], r["im_id"]) for r in rows})
    print(f"{len(out_rows)} rows / {n_imgs} images -> {args.out} "
          f"(mode={args.mode})", flush=True)


if __name__ == "__main__":
    main()
