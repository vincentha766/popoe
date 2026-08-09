"""Stage timing, gated on POPOE_PROFILE.

Off by default: with the env var unset every hook is a bare `yield`, so a BOP
run pays one module-level bool check per stage and nothing else. This exists to
answer "where does a detect call's wall clock go", which the per-image timing in
`bop_muse` cannot: that measures whole frames, and the interesting split
(cached query encoding vs per-frame target encoding, DINOv2 vs GeDi) is inside.

Why the CUDA sync is not optional
---------------------------------
Kernel launches are asynchronous. Timing a GPU stage with `perf_counter` and no
sync measures how long it took to *queue* the work; the real cost then lands on
whichever later stage happens to block on a result. The totals still add up to
the true wall clock, so the output looks entirely reasonable while attributing
the time to the wrong stage — the failure mode is a confident wrong answer, not
an obvious error. Sync before both edges of every stage.

Usage:
    with profiling.stage("tgt_gedi"):
        ...
    profiling.reset()                  # per detect call
    for name, secs, calls in profiling.report(): ...
"""
import os
import threading
import time
from contextlib import contextmanager

ENABLED = os.environ.get("POPOE_PROFILE", "") not in ("", "0")

_local = threading.local()


def _acc() -> dict:
    a = getattr(_local, "acc", None)
    if a is None:
        a = _local.acc = {}
    return a


def _sync() -> None:
    if not ENABLED:
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


@contextmanager
def stage(name: str):
    if not ENABLED:
        yield
        return
    _sync()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _sync()
        rec = _acc().setdefault(name, [0.0, 0])
        rec[0] += time.perf_counter() - t0
        rec[1] += 1


def reset() -> None:
    _local.acc = {}


def report():
    """[(name, seconds, calls), ...] sorted by seconds descending."""
    return sorted(((k, v[0], v[1]) for k, v in _acc().items()),
                  key=lambda r: -r[1])


def format_report(total: float | None = None) -> str:
    """One line per stage, ordered by cost.

    Indented names are NESTED inside the stage above them (`  tgt_gedi` is part
    of `target_encode`), so the column does not sum to `total` — do not add it
    up. `total` is the caller's own wall clock and is printed separately, which
    is what makes an unaccounted slice visible instead of silently absorbed."""
    rows = report()
    if not rows:
        return "  (空 — POPOE_PROFILE 没开?)"
    width = max(len(r[0]) for r in rows)
    out = []
    for name, secs, calls in rows:
        pct = f"{secs / total * 100:5.1f}%" if total else "      "
        out.append(f"  {name:<{width}}  {secs:8.2f}s  {pct}  x{calls}")
    if total:
        top = sum(s for n, s, _ in rows if not n.startswith(" "))
        out.append(f"  {'—— 顶层合计':<{width}}  {top:8.2f}s"
                   f"  {top / total * 100:5.1f}%  (墙钟 {total:.2f}s)")
    return "\n".join(out)
