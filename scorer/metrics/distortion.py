"""Distortion: 2-function fusion — contour deviation + Hough global voting."""

import numpy as np
import cv2
from ..preprocess.pipeline import PreprocessedImage
from ..metrics import MetricResult, register


@register
def distortion(pp: PreprocessedImage) -> MetricResult:
    """Score lens distortion via 2-function fusion.

    Sub-functions:
      1. Contour straightness (60%) — line-fit deviation per long edge
      2. Hough global voting (40%) — detects systematic line curvature pattern
    """
    y = pp.y
    h, w = y.shape
    y_u8 = (np.clip(y, 0.0, 1.0) * 255).astype(np.uint8)

    edges_canny = cv2.Canny(y_u8, 60, 180)
    edges = pp.edge_mask | (edges_canny > 0) if pp.edge_mask.sum() > 10 else edges_canny > 0

    # ── Sub-function 1: Contour deviation (60%) ──
    contours, _ = cv2.findContours(edges.astype(np.uint8), cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    min_len = max(min(w, h) // 10, 20)
    long_contours = [c for c in contours if cv2.arcLength(c, False) > min_len]

    if len(long_contours) < 3:
        heatmap = _radial_heatmap(h, w)
        return MetricResult(
            name="distortion", global_score=80.0, heatmap=heatmap,
            region_scores={name: 80.0 for name in pp.roi},
            diagnosis="直线特征不足，无法可靠评估畸变",
            metadata={"contours_analyzed": len(long_contours),
                      "sub_contour": 80.0, "sub_hough": 80.0},
        )

    deviations = []
    for c in long_contours:
        if len(c) < 5:
            continue
        pts = c.reshape(-1, 2).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        dists = np.abs((pts[:, 0] - x0) * (-vy) + (pts[:, 1] - y0) * vx)
        deviations.append(float(np.mean(dists)))

    mean_dev = float(np.mean(deviations)) if deviations else 0.0
    sub_contour = max(0.0, min(100.0, 100.0 - mean_dev * 18.0))

    # ── Sub-function 2: Hough global voting (40%) ──
    sub_hough = _hough_line_curvature(edges, h, w)

    # ── Fusion ──
    score = 0.6 * sub_contour + 0.4 * sub_hough

    heatmap = _radial_heatmap(h, w)
    region_scores = {}
    for name, mask in pp.roi.items():
        w_factor = 1.5 if "corner" in name else (1.0 if "edge" in name else 0.3)
        region_scores[name] = float(max(0.0, min(100.0,
                                                  100.0 - mean_dev * w_factor * 20.0)))

    parts = []
    if score < 40:
        parts.append(f"检测到明显畸变 (综合:{score:.0f}, 轮廓偏离:{mean_dev:.1f}px)")
    elif score < 65:
        parts.append(f"检测到轻微畸变 (综合:{score:.0f})")
    elif mean_dev > 1.0:
        parts.append(f"边缘直线度略有偏差 ({mean_dev:.1f}px)")

    diagnosis = "；".join(parts) if parts else "未检测到明显畸变"

    return MetricResult(
        name="distortion", global_score=round(score, 1), heatmap=heatmap,
        region_scores=region_scores, diagnosis=diagnosis,
        metadata={"contours_analyzed": len(long_contours),
                  "sub_contour": round(sub_contour, 1),
                  "sub_hough": round(sub_hough, 1),
                  "mean_deviation_px": round(mean_dev, 3)},
    )


def _hough_line_curvature(edges: np.ndarray, h: int, w: int) -> float:
    """Score straightness via Hough line voting pattern.

    In a distortion-free image, Hough lines cluster around a few dominant
    angles (horizontal/vertical). Under barrel/pincushion distortion,
    Hough votes spread across angles because straight lines appear curved.
    We measure the angular concentration of Hough votes.
    """
    lines = cv2.HoughLines(edges.astype(np.uint8), 1, np.pi / 180, threshold=60)
    if lines is None or len(lines) < 4:
        return 70.0  # insufficient data → neutral

    # Collect all line angles (in degrees, normalized to [0, 180))
    angles = []
    for line in lines:
        rho, theta = line[0]
        deg = (theta * 180 / np.pi) % 180
        angles.append(deg)

    angles = np.array(angles)
    if len(angles) < 4:
        return 70.0

    # Build histogram of angles (10° bins)
    hist, _ = np.histogram(angles, bins=18, range=(0, 180))
    hist = hist.astype(np.float32)
    hist = hist / hist.sum()

    # Entropy of angle distribution (lower = more concentrated = less distortion)
    eps = 1e-8
    entropy = -np.sum(hist * np.log2(hist + eps))

    # Max entropy for 18 bins ≈ 4.17 (uniform). Min ≈ 0 (all in one bin)
    # Score: entropy < 3.0 = concentrated lines = good
    return max(0.0, min(100.0, 100.0 - max(0, entropy - 2.5) * 35.0))


def _radial_heatmap(h: int, w: int) -> np.ndarray:
    cy, cx = h / 2, w / 2
    ys, xs = np.ogrid[:h, :w]
    r = np.sqrt(((xs - cx) / max(cx, 1)) ** 2 + ((ys - cy) / max(cy, 1)) ** 2)
    rmax = r.max()
    return (r / rmax).astype(np.float32) if rmax > 0 else r.astype(np.float32)
