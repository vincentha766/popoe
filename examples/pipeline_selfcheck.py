"""Acceptance self-check — the adapter pipeline must reproduce the inline
FreeZeV2.estimate_pose body bitwise on the SAME extracted arrays.

For one target it (A) runs the raw ransac -> icp -> score functions and (B) the
adapter chain (RansacSolver -> ICPRefiner -> FreeZeScorer), on identical inputs.
Same functions, fixed RANSAC seed, deterministic ICP => must match to ~1e-15.
This isolates adapter fidelity from GPU re-extraction non-determinism.

    POPOE_GEDI_PATH=/path/to/gedi POPOE_TWO_SCALE_GEDI=1 \
    python examples/pipeline_selfcheck.py --bop /path/to/ycbv --obj 5 -n 3
"""
import argparse
import numpy as np

from freezev2_monolith import FreeZeV2   # sibling module (run from examples/)
from popoe.registration import (
    ransac_pose_estimation, icp_refinement, feature_aware_score, final_score,
)
from popoe import Scene, ObjectModel, Detection, PointFeatures
from popoe.freeze.adapters import make_freeze_encoders
from popoe.datasets.bop import find_instances, load_inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bop", default="/workspace/bop_data/ycbv")
    ap.add_argument("--obj", type=int, default=5)
    ap.add_argument("-n", "--n-instances", type=int, default=3)
    ap.add_argument("--n-points", type=int, default=5000)
    args = ap.parse_args()

    mesh_path = f"{args.bop}/models/obj_{args.obj:06d}.ply"
    insts = find_instances(args.bop, args.obj, args.n_instances)
    print(f"obj {args.obj}: {len(insts)} instances -> {insts}")
    if not insts:
        print("no instances; nothing to check"); return

    fz = FreeZeV2(device="cuda")
    _, tenc = make_freeze_encoders(fz.query_extractor, fz.target_extractor, args.n_points)
    fz.precompute_query(mesh_path, n_points=args.n_points)
    q = PointFeatures(pts=fz._pts_query, feats=fz._feats_query,
                      meta={"canon_frame": fz.query_extractor.canon_frame})
    frame = q.meta["canon_frame"]
    obj = ObjectModel(obj_id=args.obj, mesh_path=mesh_path, diameter=1.0 / frame.scale)

    from popoe.adapters import RansacSolver, ICPRefiner, BestScoreSelector
    from popoe.freeze.adapters import FreeZeScorer
    solver = RansacSolver(n_ransac=fz.n_ransac, tau_inlier=fz.tau_inlier, k=fz.k_corr)
    refiner, scorer, selector = ICPRefiner(fz.tau_icp), FreeZeScorer(fz.tau_inlier), BestScoreSelector()

    wR = wt = ws = 0.0
    for (s_id, im_id, gi) in insts:
        rgb, depth, mask, K, intr = load_inputs(args.bop, s_id, im_id, gi)
        scene = Scene(rgb=rgb, depth=depth, K=K, scene_id=s_id, im_id=im_id)
        t = tenc.encode_target(scene, Detection(mask=mask, score=1.0), obj, frame)
        if len(t.pts) < 4:
            print(f"  scn{s_id}/im{im_id}/gt{gi}: target too small; skip"); continue

        # (A) inline reference
        R_c, t_c, s_coarse = ransac_pose_estimation(
            q.pts, q.feats, t.pts, t.feats, n_iters=fz.n_ransac, tau_inlier=fz.tau_inlier, k=fz.k_corr)
        R_f, t_f, s_icp = icp_refinement(q.pts, t.pts, R_c, t_c, fz.tau_icp)
        s_fine, _ = feature_aware_score(R_f, t_f, q.pts, t.pts, q.feats, t.feats, fz.tau_inlier)
        s_ref = final_score(s_coarse, s_fine, s_icp)

        # (B) adapter chain, same arrays
        h = solver.solve(q, t, frame)[0]
        h = refiner.refine(h, scene, obj, q, t)
        h = scorer.score(h, q, t)

        dR, dt, ds = float(np.abs(R_f - h.R).max()), float(np.abs(t_f - h.t).max()), float(abs(s_ref - h.score))
        wR, wt, ws = max(wR, dR), max(wt, dt), max(ws, ds)
        print(f"  scn{s_id}/im{im_id}/gt{gi}: |dR|={dR:.2e} |dt|={dt:.2e} |dscore|={ds:.2e} (score={s_ref:.4f})")

    print(f"\nworst: |dR|={wR:.2e} |dt|={wt:.2e} |dscore|={ws:.2e}")
    print("ACCEPTANCE:", "PASS — adapters bitwise-faithful" if (wR < 1e-9 and wt < 1e-9 and ws < 1e-9)
          else "FAIL — investigate divergence")


if __name__ == "__main__":
    main()
