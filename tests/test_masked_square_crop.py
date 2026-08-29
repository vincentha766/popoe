import numpy as np
import pytest

from popoe.segmentor_cnos_lab import square_crop


def _rgb(h=480, w=640):
    return np.full((h, w, 3), 200, dtype=np.uint8)


def _mask(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


# ── segmentor_cnos_lab.square_crop ─────────────────────────────────────────
# Same defect, same fix: each side was clamped independently, so an object near
# an edge produced a non-square window that .resize((size, size)) then squashed.

def _lab_window(h_img, w_img, mask, pad=0.1):
    """Mirror square_crop's window geometry so the shape can be asserted before
    the resize hides it (the function always returns size x size)."""
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = max(y1 - y0, x1 - x0)
    half = int(round(side * (0.5 + pad)))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    win = min(2 * half, h_img, w_img)
    Y0 = int(np.clip(cy - win // 2, 0, h_img - win))
    X0 = int(np.clip(cx - win // 2, 0, w_img - win))
    return win, mask[Y0:Y0 + win, X0:X0 + win].sum(), mask.sum()


@pytest.mark.parametrize(
    "h,w,box",
    [
        (480, 640, (200, 300, 270, 370)),
        (480, 640, (400, 480, 270, 370)),
        (480, 640, (200, 300, 560, 640)),
        (480, 640, (430, 480, 600, 640)),
        (480, 640, (0, 60, 0, 60)),
        (720, 1280, (600, 720, 600, 700)),
    ],
)
def test_lab_window_is_square_and_contains_mask(h, w, box):
    mask = _mask(h, w, *box)
    win, kept, total = _lab_window(h, w, mask)
    assert kept == total, "window clipped part of the mask"
    assert win > 0


def test_lab_square_crop_returns_requested_size():
    h, w = 480, 640
    crop, crop_mask = square_crop(_rgb(h, w), _mask(h, w, 400, 480, 270, 370), size=224)
    assert crop.shape[:2] == (224, 224)
    assert crop_mask.shape[:2] == (224, 224)
