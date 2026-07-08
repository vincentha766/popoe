"""
6D pose visualization utilities.
Draws 3D bounding box and axes on RGB image given estimated pose.
"""

import numpy as np
import cv2


def project_points(pts_3d, R, t, K):
    """Project 3D points to 2D image plane.
    pts_3d: (N, 3) in object/model frame
    R: (3, 3), t: (3,), K: (3, 3)
    Returns: (N, 2) pixel coords
    """
    pts_cam = (R @ pts_3d.T).T + t          # (N, 3) in camera frame
    pts_cam = pts_cam[pts_cam[:, 2] > 0]    # keep in front
    if len(pts_cam) == 0:
        return np.zeros((0, 2))
    uvw = (K @ pts_cam.T).T                 # (N, 3)
    return uvw[:, :2] / uvw[:, 2:3]        # (N, 2)


def draw_axes(img, R, t, K, length=0.05, thickness=2):
    """Draw XYZ axes at object origin. length in metres."""
    origin = np.zeros((1, 3))
    axes = np.array([[length, 0, 0], [0, length, 0], [0, 0, length]])

    o2d = project_points(origin, R, t, K)
    if len(o2d) == 0:
        return img
    ox, oy = int(o2d[0, 0]), int(o2d[0, 1])

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X=red, Y=green, Z=blue
    for i, (axis, color) in enumerate(zip(axes, colors)):
        tip = project_points(axis[None], R, t, K)
        if len(tip) == 0:
            continue
        tx, ty = int(tip[0, 0]), int(tip[0, 1])
        cv2.arrowedLine(img, (ox, oy), (tx, ty), color, thickness, tipLength=0.3)

    return img


def draw_bbox_3d(img, model_pts, R, t, K, color=(0, 255, 255), thickness=2):
    """
    Draw the 3D bounding box of the object model projected into the image.
    model_pts: (N, 3) point cloud of the object in model frame (metres)
    """
    # Compute axis-aligned bbox corners in model frame
    mins = model_pts.min(0)
    maxs = model_pts.max(0)
    corners = np.array([
        [mins[0], mins[1], mins[2]],
        [maxs[0], mins[1], mins[2]],
        [maxs[0], maxs[1], mins[2]],
        [mins[0], maxs[1], mins[2]],
        [mins[0], mins[1], maxs[2]],
        [maxs[0], mins[1], maxs[2]],
        [maxs[0], maxs[1], maxs[2]],
        [mins[0], maxs[1], maxs[2]],
    ])

    pts2d = project_points(corners, R, t, K)
    if len(pts2d) < 8:
        return img

    pts2d = pts2d.astype(int)
    edges = [
        (0,1),(1,2),(2,3),(3,0),  # bottom face
        (4,5),(5,6),(6,7),(7,4),  # top face
        (0,4),(1,5),(2,6),(3,7),  # verticals
    ]
    for i, j in edges:
        cv2.line(img, tuple(pts2d[i]), tuple(pts2d[j]), color, thickness)

    return img


def draw_mask(img, mask, color=(0, 200, 100), alpha=0.4):
    """Overlay a boolean mask on the image."""
    overlay = img.copy()
    overlay[mask] = np.array(color, dtype=np.uint8)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def visualize_pose(rgb, T_est, model_pts, K,
                   mask=None, T_gt=None, score=None,
                   draw_axes_len=0.05):
    """
    Full pose visualization. Returns annotated RGB image (uint8).

    rgb:        (H, W, 3) uint8
    T_est:      (4, 4) estimated pose (camera ← object)
    model_pts:  (N, 3) model point cloud in metres
    K:          (3, 3) camera intrinsics matrix
    mask:       (H, W) bool, optional — drawn as green overlay
    T_gt:       (4, 4) ground-truth pose, optional — drawn in a different color
    score:      float, optional — shown in corner
    """
    vis = rgb.copy()

    if mask is not None:
        vis = draw_mask(vis, mask)

    R_est = T_est[:3, :3]
    t_est = T_est[:3, 3]
    vis = draw_bbox_3d(vis, model_pts, R_est, t_est, K, color=(0, 255, 255))
    vis = draw_axes(vis, R_est, t_est, K, length=draw_axes_len)

    if T_gt is not None:
        R_gt = T_gt[:3, :3]
        t_gt = T_gt[:3, 3]
        vis = draw_bbox_3d(vis, model_pts, R_gt, t_gt, K,
                           color=(255, 100, 0), thickness=1)

    if score is not None:
        text = f"score={score:.3f}"
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return vis


def save_visualization(path, rgb, T_est, model_pts, K, **kwargs):
    """Save pose visualization to file."""
    vis = visualize_pose(rgb, T_est, model_pts, K, **kwargs)
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), vis_bgr)
    return vis
