"""Sharpness: 3-function fusion — Laplacian + edge width + gradient magnitude."""

import numpy as np
import cv2
from scipy.optimize import curve_fit
from ..preprocess.pipeline import PreprocessedImage
from ..metrics import MetricResult, register


@register
def sharpness(pp: PreprocessedImage) -> MetricResult:
    """Score sharpness via 3-function weighted fusion.

    Sub-functions:
      1. Laplacian variance (40%) — classic focus measure, sensitive to fine detail
      2. Edge width from gradient spread (30%) — simplified MTF-like measurement
      3. Gradient magnitude mean (30%) — overall edge energy
    """
    y = pp.y
    is_uniform = float(y.std()) < 0.01
    y_u8 = (np.clip(y, 0.0, 1.0) * 255).astype(np.uint8)

    # Gradient
    grad = pp.gradient_mag
    if np.allclose(grad, 0):
        gx = cv2.Sobel(y_u8, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(y_u8, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx ** 2 + gy ** 2)

    # ── Sub-function 1: Laplacian variance (40%) ──
    lap = cv2.Laplacian(y_u8, cv2.CV_32F)
    lap_var = float(lap.var())
    sub_lap = min(100.0, lap_var / 50.0) if not is_uniform else 85.0

    # ── Sub-function 2: Edge width (30%) ──
    edge_width = _estimate_edge_width(grad) if not is_uniform else 0.5
    # edge_width ~0.5px = sharp, >5px = blurry
    sub_edge = max(0.0, min(100.0, 100.0 - (edge_width - 0.5) * 18.0)) if not is_uniform else 90.0

    # ── Sub-function 3: Gradient magnitude mean (30%) ──
    mean_grad = float(grad.mean())
    # mean_grad ~0.05 for sharp image, ~0.005 for blurry
    sub_grad = min(100.0, mean_grad / 0.05 * 100.0) if not is_uniform else 85.0

    # ── Fusion ──
    if is_uniform:
        score = 85.0
    else:
        score = 0.4 * sub_lap + 0.3 * sub_edge + 0.3 * sub_grad

    # Heatmap: inverted gradient (darker = blurrier region)
    heatmap = 1.0 / (grad + 1.0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = (heatmap * (100 - score) / 100.0).astype(np.float32)

    # Region scores (use Laplacian for consistency)
    region_scores = {}
    for name, mask in pp.roi.items():
        if is_uniform:
            region_scores[name] = 85.0
        else:
            lr = lap[mask]
            region_scores[name] = float(min(100.0, lr.var() / 50.0)) if lr.size > 0 else 0.0

    # Diagnosis
    parts = []
    if is_uniform:
        parts.append("均匀画面，清晰度不适用")
    else:
        center_score = region_scores.get("center", 0)
        edge_scores = [v for k, v in region_scores.items() if "edge" in k]
        corner_scores = [v for k, v in region_scores.items() if "corner" in k]
        if center_score < 60:
            parts.append(f"中心区域清晰度不足 (得分{center_score:.1f})")
        if edge_scores and np.mean(edge_scores) < 60:
            parts.append(f"边缘区域清晰度不足 (平均{np.mean(edge_scores):.1f})")
        if corner_scores and np.mean(corner_scores) < 50:
            parts.append(f"角落区域清晰度明显下降 (平均{np.mean(corner_scores):.1f})")
        if sub_edge < 40:
            parts.append(f"边缘宽度偏大 (edge_score={sub_edge:.0f})，疑似模糊")

    diagnosis = "；".join(parts) if parts else "清晰度正常"

    return MetricResult(
        name="sharpness",
        global_score=round(score, 1),
        heatmap=heatmap,
        region_scores=region_scores,
        diagnosis=diagnosis,
        metadata={"sub_lap_var": round(sub_lap, 1), "sub_edge_width": round(sub_edge, 1),
                  "sub_grad_mag": round(sub_grad, 1),
                  "edge_width_px": round(edge_width, 3)},
    )


def _estimate_edge_width(grad: np.ndarray) -> float:
    gx_proj = grad.sum(axis=0)
    if gx_proj.sum() < 1e-6:
        return 10.0
    gx_proj = gx_proj / gx_proj.sum()
    cols = np.arange(len(gx_proj))
    center = np.sum(cols * gx_proj)
    spread = np.sqrt(np.sum((cols - center) ** 2 * gx_proj))
    return float(spread)
