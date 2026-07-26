"""Pluggability demo — run the SAME pipeline with different PoseSolver
implementations by changing ONE line, and score each against GT.

    solver = RansacSolver(...)                       # hand-rolled feature-aware RANSAC
    solver = Open3DFeatureRansacSolver(n_restarts=1)  # Open3D C++ RANSAC, 1 shot
    solver = Open3DFeatureRansacSolver(n_restarts=8)  # + feature-aware re-ranking

Everything downstream (ICP refiner, scorer, selector) is identical, so this shows
the stage is swappable.

Ranking is by **MSSD** — bop_toolkit's own symmetry-aware surface distance, with
symmetries expanded from `models_eval/models_info.json`. That is not a detail:
the usual subject (YCB-V obj 5, the mustard bottle) is near-symmetric, and an
earlier A/B here was withdrawn precisely because it ranked on a raw geodesic
rotation distance. Under that metric ~half of all instances sit in a 180°-flipped
mode for EVERY solver, the median lands on a bimodal boundary, and a 3-point
change in flip rate swings it 125° (see ARCHITECTURE.md, Pluggability, and
ISSUES.md 2026-07-26). The rotation/translation numbers are still printed, but
only for continuity — never rank a symmetric object on them.

Needs bop_toolkit (`POPOE_BOP_TOOLKIT`); it raises rather than falling back to
the symmetry-blind metric, since that substitution is the whole original defect.

Pass --seed for a reproducible run: Open3D's RANSAC is otherwise unseeded.

    POPOE_GEDI_PATH=/path/to/gedi POPOE_TWO_SCALE_GEDI=1 \
    python -u examples/solver_swap_demo.py --bop /path/to/ycbv --obj 5 -n 150 --seed 42
"""
import argparse
import json
import os

import numpy as np

from freezev2_monolith import FreeZeV2   # sibling module (run from examples/)
from popoe import Scene, ObjectModel, Detection, PointFeatures
from popoe.adapters import RansacSolver, ICPRefiner, BestScoreSelector
from popoe.freeze.adapters import make_freeze_encoders, FreeZeScorer
from popoe.solvers import Open3DFeatureRansacSolver
from popoe.datasets.bop import find_instances, load_inputs, load_gt


def load_eval_model(bop_root, obj_id):
    """models_eval vertices (mm) + BOP symmetry transformations + diameter.

    Uses bop_toolkit's own symmetry expansion and, below, its own MSSD, so the
    ranking metric here is the reference implementation rather than a local
    re-derivation. Missing toolkit raises: a silent fall back to the
    symmetry-BLIND rotation error is what made the withdrawn A/B meaningless.
    """
    import sys
    sys.path.insert(0, os.environ.get("POPOE_BOP_TOOLKIT", "/workspace/bop_toolkit"))
    from bop_toolkit_lib import misc
    import trimesh

    info = json.load(open(f"{bop_root}/models_eval/models_info.json"))[str(obj_id)]
    mesh = trimesh.load(f"{bop_root}/models_eval/obj_{obj_id:06d}.ply")
    syms = misc.get_symmetry_transformations(info, max_sym_disc_step=0.01)
    return np.asarray(mesh.vertices, np.float64), syms, float(info["diameter"])


def pose_err(R, t_m, R_gt, t_gt_mm, pts, syms):
    """(MSSD mm, symmetry-blind rot deg, trans mm).

    MSSD is the number that ranks solvers; the other two are printed for
    continuity with the old output and must NOT be used to rank a symmetric
    object — see this module's docstring.
    """
    from bop_toolkit_lib import pose_error
    t_est = (np.asarray(t_m, np.float64) * 1000.0).reshape(3, 1)
    t_gt = np.asarray(t_gt_mm, np.float64).reshape(3, 1)
    mssd = float(pose_error.mssd(R, t_est, R_gt, t_gt, pts, syms))
    dt = float(np.linalg.norm(t_est - t_gt))
    cos = (np.trace(R_gt.T @ R) - 1.0) / 2.0
    return mssd, float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))), dt


def run_chain(solver, refiner, scorer, selector, q, t, frame, scene, obj):
    hyps = solver.solve(q, t, frame)
    if not hyps:
        return None
    cands = [scorer.score(refiner.refine(h, scene, obj, q, t), q, t) for h in hyps]
    return selector.select(cands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bop", default="/workspace/bop_data/ycbv")
    ap.add_argument("--obj", type=int, default=5)
    ap.add_argument("-n", "--n-instances", type=int, default=5)
    ap.add_argument("--n-points", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=None,
                    help="seed Open3D's global RANSAC RNG; without it the two "
                         "open3d rows are not reproducible run-to-run")
    args = ap.parse_args()

    mesh_path = f"{args.bop}/models/obj_{args.obj:06d}.ply"
    insts = find_instances(args.bop, args.obj, args.n_instances)
    pts_eval, syms, diameter = load_eval_model(args.bop, args.obj)
    print(f"obj {args.obj}: {len(insts)} instances, "
          f"diameter {diameter:.1f}mm, {len(syms)} symmetry transform(s)")

    fz = FreeZeV2(device="cuda")
    _, tenc = make_freeze_encoders(fz.query_extractor, fz.target_extractor, args.n_points)
    refiner, scorer, selector = ICPRefiner(fz.tau_icp), FreeZeScorer(fz.tau_inlier), BestScoreSelector()
    solvers = {
        "freeze_ransac": RansacSolver(n_ransac=fz.n_ransac, tau_inlier=fz.tau_inlier, k=fz.k_corr),
        "open3d_1shot": Open3DFeatureRansacSolver(tau_inlier=fz.tau_inlier, n_restarts=1,
                                                  seed=args.seed),
        "open3d_rerank": Open3DFeatureRansacSolver(tau_inlier=fz.tau_inlier, n_restarts=8,
                                                   seed=args.seed),
    }
    print(f"instances={args.n_instances} seed={args.seed} "
          f"({'reproducible' if args.seed is not None else 'UNSEEDED — open3d rows will vary'})")

    fz.precompute_query(mesh_path, n_points=args.n_points)
    q = PointFeatures(pts=fz._pts_query, feats=fz._feats_query,
                      meta={"canon_frame": fz.query_extractor.canon_frame})
    frame = q.meta["canon_frame"]
    obj = ObjectModel(obj_id=args.obj, mesh_path=mesh_path, diameter=1.0 / frame.scale)

    names = list(solvers)
    def fmt(v):
        return "     (no solution)    " if v is None else \
               f"{v[0]:7.1f}mm/{v[1]:6.1f}deg"
    hdr = f"\n{'instance':>16} | " + " | ".join(f"{n:>21}" for n in names)
    print(hdr); print("-" * len(hdr))
    agg = {k: [] for k in solvers}
    for (s_id, im_id, gi) in insts:
        rgb, depth, mask, K, intr = load_inputs(args.bop, s_id, im_id, gi)
        R_gt, t_gt = load_gt(args.bop, s_id, im_id, gi)
        scene = Scene(rgb=rgb, depth=depth, K=K, scene_id=s_id, im_id=im_id)
        t = tenc.encode_target(scene, Detection(mask=mask, score=1.0), obj, frame)
        row = {}
        for name, solver in solvers.items():
            best = run_chain(solver, refiner, scorer, selector, q, t, frame, scene, obj)
            row[name] = None if best is None else pose_err(
                best.R, best.t, R_gt, t_gt, pts_eval, syms)
            if row[name]:
                agg[name].append(row[name])
        print(f"  scn{s_id}/im{im_id}/gt{gi:>2} | " + " | ".join(f"{fmt(row[n]):>21}" for n in names))

    print(f"\n=== MSSD, symmetry-aware (lower = better); diameter {diameter:.1f}mm ===")
    for name, errs in agg.items():
        if not errs:
            continue
        m = np.array([e[0] for e in errs])
        rec = [(m < f * diameter).mean() for f in (0.05, 0.10, 0.20)]
        print(f"  {name:>16}: median {np.median(m):7.1f}mm ({np.median(m)/diameter:.3f}d)  "
              f"recall@0.05d {rec[0]:.3f} @0.1d {rec[1]:.3f} @0.2d {rec[2]:.3f}  "
              f"({len(errs)}/{len(insts)})")
    print("\n--- symmetry-BLIND rotation, for continuity only; do NOT rank on this ---")
    for name, errs in agg.items():
        if errs:
            print(f"  {name:>16}: median rot {np.median([e[1] for e in errs]):6.2f}deg  "
                  f"trans {np.median([e[2] for e in errs]):7.1f}mm")
    print("\nPluggability: three PoseSolver implementations, one pipeline, one line changed.")


if __name__ == "__main__":
    main()
