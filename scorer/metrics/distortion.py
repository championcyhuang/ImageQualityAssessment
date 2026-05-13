"""Lens distortion metric — measures edge straightness via contour line-fitting."""

import numpy as np
import cv2
from ..preprocess.pipeline import PreprocessedImage
from ..metrics import MetricResult, register


@register
def distortion(pp: PreprocessedImage) -> MetricResult:
    """Score lens distortion by analyzing edge straightness.

    Detects long contours in the edge map, fits straight lines,
    and measures average point-to-line deviation. Higher deviation
    indicates stronger barrel/pincushion distortion (curved lines).
    Heatmap reflects radial distortion profile (strongest at corners).
    """
    y = pp.y
    h, w = y.shape
    y_u8 = (np.clip(y, 0.0, 1.0) * 255).astype(np.uint8)

    # Combine gradient edges + Canny for better coverage
    edges_canny = cv2.Canny(y_u8, 60, 180)
    if pp.edge_mask.sum() > 10:
        edges = pp.edge_mask | (edges_canny > 0)
    else:
        edges = edges_canny > 0

    # Find contours
    contours, _ = cv2.findContours(edges.astype(np.uint8), cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Filter for long contours (likely real structural edges)
    min_len = max(min(w, h) // 10, 20)
    long_contours = [c for c in contours if cv2.arcLength(c, False) > min_len]

    if len(long_contours) < 3:
        heatmap = _radial_heatmap(h, w)
        return MetricResult(
            name="distortion",
            global_score=80.0,
            heatmap=heatmap,
            region_scores={name: 80.0 for name in pp.roi},
            diagnosis="直线特征不足，无法可靠评估畸变",
            metadata={"contours_analyzed": len(long_contours),
                      "mean_deviation_px": 0.0},
        )

    # Fit line to each contour, measure average deviation
    deviations = []
    for c in long_contours:
        if len(c) < 5:
            continue
        pts = c.reshape(-1, 2).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        # Point-to-line distance: |(pt - line_origin) x line_direction|
        dists = np.abs((pts[:, 0] - x0) * (-vy) + (pts[:, 1] - y0) * vx)
        deviations.append(float(np.mean(dists)))

    mean_dev = float(np.mean(deviations)) if deviations else 0.0

    # Score: ~0.5px deviation = excellent, >3px = poor
    score = max(0.0, min(100.0, 100.0 - mean_dev * 18.0))

    # Heatmap: radial profile (distortion increases from center outward)
    heatmap = _radial_heatmap(h, w)

    # Per-region scores (corners have higher distortion sensitivity)
    region_scores = {}
    for name, mask in pp.roi.items():
        w_factor = 1.5 if "corner" in name else (1.0 if "edge" in name else 0.3)
        region_scores[name] = float(max(0.0, min(100.0,
                                                  100.0 - mean_dev * w_factor * 20.0)))

    # Diagnosis
    parts = []
    if mean_dev > 2.5:
        parts.append(f"检测到明显畸变 (轮廓平均偏离 {mean_dev:.1f}px)")
    elif mean_dev > 1.2:
        parts.append(f"检测到轻微畸变 (轮廓平均偏离 {mean_dev:.1f}px)")
    elif mean_dev > 0.6:
        parts.append(f"边缘直线度略有偏差 ({mean_dev:.1f}px)")

    diagnosis = "；".join(parts) if parts else "未检测到明显畸变"

    return MetricResult(
        name="distortion",
        global_score=round(score, 1),
        heatmap=heatmap,
        region_scores=region_scores,
        diagnosis=diagnosis,
        metadata={"contours_analyzed": len(long_contours),
                  "mean_deviation_px": round(mean_dev, 3)},
    )


def _radial_heatmap(h: int, w: int) -> np.ndarray:
    """Normalized radial map: 0 at center, 1 at farthest corner."""
    cy, cx = h / 2, w / 2
    ys, xs = np.ogrid[:h, :w]
    r = np.sqrt(((xs - cx) / max(cx, 1)) ** 2 + ((ys - cy) / max(cy, 1)) ** 2)
    rmax = r.max()
    return (r / rmax).astype(np.float32) if rmax > 0 else r.astype(np.float32)
