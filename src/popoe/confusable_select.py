"""Confusable-pair selection: dual-CAD assignment for same-shape / different-size objects.

The formal per-object path scores each (object, mask) independently with
``ChampionScorer(size_aware=True)``. That is **not** dual-model assignment:
when two clamps co-occur, both CADs can claim the same instance and
``metric_fit`` alone can lock the larger CAD onto the smaller one.

This module assigns each shared candidate index to at most one object by
comparing ``metric_fit`` (tie-break: scale-blind product), then each object
picks its best remaining candidate under the size-aware product rule.

Used by:
  * offline ``scripts/eval_dual_cad_metric_fit_ab.py`` / BOP CSV export
  * ``examples/bop_eval.py --dual-assign`` (buffers confusable objects per image)

Cand-index alignment assumes both objects were segmented from the **same**
ordered merge pool (same ``merge_labels`` + top-K ordering).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence


def product_score(
    s_icp: float,
    s_feat_1: float,
    metric_fit: float = 1.0,
    *,
    use_metric_fit: bool = True,
) -> float:
    s = max(float(s_icp), 0.0) * max(float(s_feat_1), 0.0)
    if use_metric_fit:
        s *= max(float(metric_fit), 0.0)
    return s


def score_from_row(row: Mapping, *, use_metric_fit: bool = True) -> float:
    return product_score(
        float(row["s_icp"]),
        float(row["s_feat_1"]),
        float(row.get("metric_fit", 1.0)),
        use_metric_fit=use_metric_fit,
    )


def score_from_hyp(hyp, *, use_metric_fit: bool = True) -> float:
    """Score a ``PoseHypothesis`` using breakdown fields (live path)."""
    bd = getattr(hyp, "breakdown", {}) or {}
    return product_score(
        float(bd.get("s_icp", bd.get("fitness", 0.0))),
        float(bd.get("s_feat_1", 0.0)),
        float(bd.get("metric_fit", 1.0)),
        use_metric_fit=use_metric_fit,
    )


def collapse_best_per_cand(
    rows: Sequence[Mapping],
    *,
    use_metric_fit: bool = True,
) -> dict[int, Mapping]:
    """Keep the best weight (or hyp) per cand index."""
    best: dict[int, Mapping] = {}
    for r in rows:
        c = int(r["cand"])
        s = score_from_row(r, use_metric_fit=use_metric_fit)
        if c not in best or s > score_from_row(best[c], use_metric_fit=use_metric_fit):
            best[c] = r
    return best


def collapse_best_per_cand_hyps(
    hyps_by_det: Mapping[int, Sequence],
    *,
    use_metric_fit: bool = True,
) -> dict[int, object]:
    """``{cand_idx: best PoseHypothesis}`` from live ``hyps_by_det``."""
    best: dict[int, object] = {}
    for ci, hs in hyps_by_det.items():
        for h in hs:
            if h is None:
                continue
            s = score_from_hyp(h, use_metric_fit=use_metric_fit)
            if ci not in best or s > score_from_hyp(best[ci], use_metric_fit=use_metric_fit):
                best[ci] = h
    return best


@dataclass(frozen=True)
class DualPick:
    """Result of dual assignment for one query object."""
    cand: Optional[int]
    row_or_hyp: object  # Mapping row or PoseHypothesis
    assigned_cands: tuple[int, ...]  # cand indices kept for this object
    fallback: bool  # True if partner missing / no shared cands → independent


def dual_assign_rows(
    query_rows: Sequence[Mapping],
    partner_rows: Sequence[Mapping],
    *,
    use_metric_fit: bool = True,
) -> DualPick:
    """Pick the best row for the query object under dual-CAD assignment.

    For each cand index present under **both** objects, assign to the CAD with
    higher ``metric_fit`` (tie → higher scale-blind product). Query-only cands
    stay with the query. Then pick argmax size-aware product among query's
    assigned set. If the partner set is empty or no assignment remains, fall
    back to independent size-aware argmax over ``query_rows``.
    """
    if not query_rows:
        return DualPick(None, None, (), fallback=True)

    if not partner_rows:
        best = max(query_rows, key=lambda r: score_from_row(r, use_metric_fit=use_metric_fit))
        return DualPick(int(best["cand"]), best, (int(best["cand"]),), fallback=True)

    q_best = collapse_best_per_cand(query_rows, use_metric_fit=use_metric_fit)
    p_best = collapse_best_per_cand(partner_rows, use_metric_fit=use_metric_fit)
    shared = sorted(set(q_best) & set(p_best))

    assigned: list[Mapping] = []
    assigned_ids: list[int] = []
    for c in shared:
        rq, rp = q_best[c], p_best[c]
        mf_q = float(rq.get("metric_fit", 1.0))
        mf_p = float(rp.get("metric_fit", 1.0))
        if mf_q > mf_p + 1e-9:
            assigned.append(rq)
            assigned_ids.append(c)
        elif mf_p > mf_q + 1e-9:
            continue
        else:
            if score_from_row(rq, use_metric_fit=False) >= score_from_row(
                rp, use_metric_fit=False
            ):
                assigned.append(rq)
                assigned_ids.append(c)

    for c, rq in q_best.items():
        if c not in p_best:
            assigned.append(rq)
            assigned_ids.append(c)

    if not assigned:
        best = max(query_rows, key=lambda r: score_from_row(r, use_metric_fit=use_metric_fit))
        return DualPick(int(best["cand"]), best, (int(best["cand"]),), fallback=True)

    best = max(assigned, key=lambda r: score_from_row(r, use_metric_fit=use_metric_fit))
    return DualPick(
        int(best["cand"]),
        best,
        tuple(sorted(set(assigned_ids))),
        fallback=False,
    )


def dual_assign_hyps(
    query_hyps_by_det: Mapping[int, Sequence],
    partner_hyps_by_det: Mapping[int, Sequence],
    *,
    use_metric_fit: bool = True,
):
    """Live dual assign over ``hyps_by_det`` maps. Returns best hyp or None."""
    if not query_hyps_by_det:
        return None
    q_best = collapse_best_per_cand_hyps(query_hyps_by_det, use_metric_fit=use_metric_fit)
    if not partner_hyps_by_det:
        return max(q_best.values(), key=lambda h: score_from_hyp(h, use_metric_fit=use_metric_fit))

    p_best = collapse_best_per_cand_hyps(partner_hyps_by_det, use_metric_fit=use_metric_fit)
    shared = sorted(set(q_best) & set(p_best))
    assigned = []
    for c in shared:
        hq, hp = q_best[c], p_best[c]
        mf_q = float((hq.breakdown or {}).get("metric_fit", 1.0))
        mf_p = float((hp.breakdown or {}).get("metric_fit", 1.0))
        if mf_q > mf_p + 1e-9:
            assigned.append(hq)
        elif mf_p > mf_q + 1e-9:
            continue
        else:
            if score_from_hyp(hq, use_metric_fit=False) >= score_from_hyp(
                hp, use_metric_fit=False
            ):
                assigned.append(hq)
    for c, hq in q_best.items():
        if c not in p_best:
            assigned.append(hq)
    if not assigned:
        return max(q_best.values(), key=lambda h: score_from_hyp(h, use_metric_fit=use_metric_fit))
    return max(assigned, key=lambda h: score_from_hyp(h, use_metric_fit=use_metric_fit))


def partner_id(obj_id: int, merge_labels: Mapping[int, Sequence[int]]) -> Optional[int]:
    """Other id in a 2-way confusable merge group, or None."""
    group = list(merge_labels.get(int(obj_id), []))
    others = [g for g in group if int(g) != int(obj_id)]
    if len(others) == 1:
        return int(others[0])
    return None
