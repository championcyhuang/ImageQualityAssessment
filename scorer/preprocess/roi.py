import numpy as np


def generate_roi_masks(width: int, height: int) -> dict[str, np.ndarray]:
    """Generate non-overlapping boolean masks for 9 ROI regions.

    center  — middle 50% width × 50% height (25% area)
    edges   — top/bot 25% height full-width; left/right 50% height × 25% width
    corners — four 25% width × 25% height squares
    """
    h_half = height // 2
    h_qtr = height // 4
    w_half = width // 2
    w_qtr = width // 4

    masks = {}

    # Center: middle 50% x 50%
    masks["center"] = _rect_mask(width, height, w_qtr, h_qtr, w_half, h_half)

    # Edges: strips between corners at top, bottom, left, right
    masks["edge_top"]    = _rect_mask(width, height, w_qtr, 0, w_half, h_qtr)
    masks["edge_bot"]    = _rect_mask(width, height, w_qtr, height - h_qtr, w_half, h_qtr)
    masks["edge_left"]   = _rect_mask(width, height, 0, h_qtr, w_qtr, h_half)
    masks["edge_right"]  = _rect_mask(width, height, width - w_qtr, h_qtr, w_qtr, h_half)

    # Corners
    masks["corner_tl"] = _rect_mask(width, height, 0, 0, w_qtr, h_qtr)
    masks["corner_tr"] = _rect_mask(width, height, width - w_qtr, 0, w_qtr, h_qtr)
    masks["corner_bl"] = _rect_mask(width, height, 0, height - h_qtr, w_qtr, h_qtr)
    masks["corner_br"] = _rect_mask(width, height, width - w_qtr, height - h_qtr, w_qtr, h_qtr)

    return masks


def _rect_mask(w: int, h: int, x: int, y: int, rw: int, rh: int) -> np.ndarray:
    """Create boolean mask for rectangle (x, y, rw, rh) in (h, w) image."""
    mask = np.zeros((h, w), dtype=bool)
    mask[y:y + rh, x:x + rw] = True
    return mask
