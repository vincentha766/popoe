"""Pluggability demo — run the SAME pipeline with different PoseSolver
implementations by changing ONE line, and score each against GT.

    solver = RansacSolver(...)                       # hand-rolled feature-aware RANSAC
    solver = Open3DFeatureRansacSolver(n_restarts=1)  # Open3D C++ RANSAC, 1 shot
    solver = Open3DFeatureRansacSolver(n_restarts=8)  # + feature-aware re-ranking

Everything downstream (ICP refiner, scorer, selector) is identical. Shows both
that the stage is swappable and how "geometry proposes, features dispose" fixes
the 1-shot flips on near-symmetric objects.

    POPOE_GEDI_PATH=/path/to/gedi POPOE_TWO_SCALE_GEDI=1 \
    python examples/solver_swap_demo.py --bop /path/to/ycbv --obj 5 -n 5
"""
import argparse
import numpy as np

from freezev2_monolith import FreeZeV2   # sibling module (run from examples/)
from popoe import Scene, ObjectModel, Detection, PointFeatures
from popoe.adapters import RansacSolver, ICPRefiner, BestScoreSelector
from popoe.freeze.adapters import make_freeze_encoders, FreeZeScorer
from popoe.solvers import Open3DFeatureRansacSolver
from popoe.datasets.bop import find_instances, load_inputs, load_gt


def pose_err(R, t_m, R_gt, t_gt_mm):
    dt = float(np.linalg.norm(t_m * 1000.0 - t_gt_mm))
    cos = (np.trace(R_gt.T @ R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))), dt


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
    args = ap.parse_args()

    mesh_path = f"{args.bop}/models/obj_{args.obj:06d}.ply"
    insts = find_instances(args.bop, args.obj, args.n_instances)
    print(f"obj {args.obj}: {len(insts)} instances")

    fz = FreeZeV2(device="cuda")
    _, tenc = make_freeze_encoders(fz.query_extractor, fz.target_extractor, args.n_points)
    refiner, scorer, selector = ICPRefiner(fz.tau_icp), FreeZeScorer(fz.tau_inlier), BestScoreSelector()
    solvers = {
        "freeze_ransac": RansacSolver(n_ransac=fz.n_ransac, tau_inlier=fz.tau_inlier, k=fz.k_corr),
        "open3d_1shot": Open3DFeatureRansacSolver(tau_inlier=fz.tau_inlier, n_restarts=1),
        "open3d_rerank": Open3DFeatureRansacSolver(tau_inlier=fz.tau_inlier, n_restarts=8),
    }

    fz.precompute_query(mesh_path, n_points=args.n_points)
    q = PointFeatures(pts=fz._pts_query, feats=fz._feats_query,
                      meta={"canon_frame": fz.query_extractor.canon_frame})
    frame = q.meta["canon_frame"]
    obj = ObjectModel(obj_id=args.obj, mesh_path=mesh_path, diameter=1.0 / frame.scale)

    names = list(solvers)
    def fmt(v): return "   (no solution)   " if v is None else f"{v[0]:7.2f}deg/{v[1]:6.1f}mm"
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
            row[name] = None if best is None else pose_err(best.R, best.t, R_gt, t_gt)
            if row[name]:
                agg[name].append(row[name])
        print(f"  scn{s_id}/im{im_id}/gt{gi:>2} | " + " | ".join(f"{fmt(row[n]):>21}" for n in names))

    print("\n=== median error (lower = better) ===")
    for name, errs in agg.items():
        if errs:
            a, d = np.median([e[0] for e in errs]), np.median([e[1] for e in errs])
            print(f"  {name:>16}: rot {a:6.2f}deg  trans {d:7.1f}mm  ({len(errs)}/{len(insts)})")
    print("\nPluggability: three PoseSolver implementations, one pipeline, one line changed.")


if __name__ == "__main__":
    main()
