"""Pluggability demo — same encoders/refiner/scorer, different PoseSolver.

    solver = Open3DFeatureRansacSolver(n_restarts=1)
    solver = Open3DFeatureRansacSolver(n_restarts=8)
    solver = GPURansacSolver()

Needs bop_toolkit (`POPOE_BOP_TOOLKIT`). Pass --seed for a reproducible o3d run.

    POPOE_GEDI_PATH=/path/to/gedi POPOE_TWO_SCALE_GEDI=1 \
    python -u examples/solver_swap_demo.py --bop /path/to/ycbv --obj 5 -n 150 --seed 42
"""
import argparse
import json
import os

import numpy as np

from popoe import Scene, ObjectModel, Detection, PointFeatures
from popoe.adapters import ICPRefiner, best_hyp
from popoe.freeze.recipes import best_encoders
from popoe.scoring import ChampionScorer
from popoe.solvers import GPURansacSolver, Open3DFeatureRansacSolver
from popoe.datasets.bop import find_instances, load_inputs, load_gt


RECALL_FRACS = (0.05, 0.10, 0.20, 0.50)


def load_eval_model(bop_root, obj_id):
    import sys
    sys.path.insert(0, os.environ.get("POPOE_BOP_TOOLKIT", "/workspace/bop_toolkit"))
    from bop_toolkit_lib import misc
    import trimesh

    info = json.load(open(f"{bop_root}/models_eval/models_info.json"))[str(obj_id)]
    mesh = trimesh.load(f"{bop_root}/models_eval/obj_{obj_id:06d}.ply")
    syms = misc.get_symmetry_transformations(info, max_sym_disc_step=0.01)
    return np.asarray(mesh.vertices, np.float64), syms, float(info["diameter"])


def pose_err(R, t_m, R_gt, t_gt_mm, pts, syms):
    from bop_toolkit_lib import pose_error
    t_est = (np.asarray(t_m, np.float64) * 1000.0).reshape(3, 1)
    t_gt = np.asarray(t_gt_mm, np.float64).reshape(3, 1)
    mssd = float(pose_error.mssd(R, t_est, R_gt, t_gt, pts, syms))
    dt = float(np.linalg.norm(t_est - t_gt))
    cos = (np.trace(R_gt.T @ R) - 1.0) / 2.0
    return mssd, float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))), dt


def run_chain(solver, refiner, scorer, q, t, scene, obj):
    hyps = solver.solve(q, t)
    if not hyps:
        return None
    cands = [scorer.score(refiner.refine(h, scene, obj, q, t), q, t) for h in hyps]
    return best_hyp(cands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bop", default="/workspace/bop_data/ycbv")
    ap.add_argument("--obj", type=int, default=5)
    ap.add_argument("-n", "--n-instances", type=int, default=5)
    ap.add_argument("--n-points", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    os.environ.setdefault("POPOE_QUERY_POINTS", str(args.n_points))
    mesh_path = f"{args.bop}/models/obj_{args.obj:06d}.ply"
    insts = find_instances(args.bop, args.obj, args.n_instances)
    pts_eval, syms, diameter = load_eval_model(args.bop, args.obj)
    print(f"obj {args.obj}: {len(insts)} instances, "
          f"diameter {diameter:.1f}mm, {len(syms)} symmetry transform(s)")

    qenc, tenc = best_encoders()
    obj = ObjectModel(obj_id=args.obj, mesh_path=mesh_path, diameter=diameter / 1000.0)
    q = qenc.encode_query(obj)
    frame = q.meta["canon_frame"]
    tau = 0.03 * (1.0 / frame.scale)
    refiner, scorer = ICPRefiner(tau_icp=tau), ChampionScorer(tau_abs=tau)
    solvers = {
        "open3d_1shot": Open3DFeatureRansacSolver(tau_inlier=tau, n_restarts=1,
                                                  seed=args.seed),
        "open3d_rerank": Open3DFeatureRansacSolver(tau_inlier=tau, n_restarts=8,
                                                   seed=args.seed),
        "gpu": GPURansacSolver(tau_inlier=tau),
    }
    print(f"instances={args.n_instances} seed={args.seed} "
          f"({'reproducible' if args.seed is not None else 'UNSEEDED — open3d rows will vary'})")

    names = list(solvers)
    def fmt(v):
        return "     (no solution)    " if v is None else \
               f"{v[0]:7.1f}mm/{v[1]:6.1f}deg"
    hdr = f"\n{'instance':>16} | " + " | ".join(f"{n:>21}" for n in names)
    print(hdr); print("-" * len(hdr))
    agg = {k: [] for k in solvers}
    for (s_id, im_id, gi) in insts:
        rgb, depth, mask, K, _intr = load_inputs(args.bop, s_id, im_id, gi)
        R_gt, t_gt = load_gt(args.bop, s_id, im_id, gi)
        scene = Scene(rgb=rgb, depth=depth, K=K, scene_id=s_id, im_id=im_id)
        t = tenc.encode_target(scene, Detection(mask=mask, score=1.0), obj, frame)
        row = {}
        for name, solver in solvers.items():
            best = run_chain(solver, refiner, scorer, q, t, scene, obj)
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
        rec = "  ".join(f"@{f:g}d {(m < f * diameter).mean():.3f}"
                        for f in RECALL_FRACS)
        print(f"  {name:>16}: median {np.median(m):7.1f}mm ({np.median(m)/diameter:.3f}d)  "
              f"recall {rec}  ({len(errs)}/{len(insts)})")
    print("\n--- symmetry-BLIND rotation, for continuity only; do NOT rank on this ---")
    for name, errs in agg.items():
        if errs:
            print(f"  {name:>16}: median rot {np.median([e[1] for e in errs]):6.2f}deg  "
                  f"trans {np.median([e[2] for e in errs]):7.1f}mm")
    print("\nPluggability: three PoseSolver implementations, one pipeline.")


if __name__ == "__main__":
    main()
