import numpy as np
from scipy.optimize import curve_fit
from ..preprocess.pipeline import PreprocessedImage
from ..metrics import MetricResult, register


@register
def sharpness(pp: PreprocessedImage) -> MetricResult:
    """Score sharpness by analyzing edge profile widths and gradient statistics.

    Uses variance of Laplacian (well-established focus measure) for global score
    and gradient magnitude for edge profile measurement. Higher = sharper.
    """
    y = pp.y
    import cv2
    y_u8 = (np.clip(y, 0.0, 1.0) * 255).astype(np.uint8)

    # If gradient is all zeros (e.g., test fixtures), compute from Y
    grad = pp.gradient_mag
    if np.allclose(grad, 0):
        gx = cv2.Sobel(y_u8, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(y_u8, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx ** 2 + gy ** 2)

    # Global sharpness: variance of Laplacian (classic focus measure)
    lap = cv2.Laplacian(y_u8, cv2.CV_32F)
    lap_var = float(lap.var())
    # lap_var ~ 5000+ for sharp binary edge, ~100 for blurred
    score = min(100.0, lap_var / 50.0)

    mean_grad = float(grad.mean())

    # Estimate edge width from gradient profile (simplified MTF-like)
    edge_width = _estimate_edge_width(grad)

    heatmap = 1.0 / (grad + 1.0)  # Low values = sharper (inverted for display)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = (heatmap * (100 - score) / 100.0).astype(np.float32)
    region_scores = {}
    for name, mask in pp.roi.items():
        lap_roi = lap[mask]
        if lap_roi.size > 0:
            roi_lap_var = float(lap_roi.var())
            region_scores[name] = float(min(100.0, roi_lap_var / 50.0))
        else:
            region_scores[name] = 0.0

    parts = []
    # Find worst ROI
    center_score = region_scores.get("center", 0)
    edge_scores = [v for k, v in region_scores.items() if "edge" in k]
    corner_scores = [v for k, v in region_scores.items() if "corner" in k]

    if center_score < 60:
        parts.append(f"中心区域清晰度不足 (得分{center_score:.1f})")
    if edge_scores and np.mean(edge_scores) < 60:
        parts.append(f"边缘区域清晰度不足 (平均{np.mean(edge_scores):.1f})")
    if corner_scores and np.mean(corner_scores) < 50:
        parts.append(f"角落区域清晰度明显下降 (平均{np.mean(corner_scores):.1f})")

    diagnosis = "；".join(parts) if parts else "清晰度正常"

    return MetricResult(
        name="sharpness",
        global_score=round(score, 1),
        heatmap=heatmap,
        region_scores=region_scores,
        diagnosis=diagnosis,
        metadata={"mean_gradient": round(mean_grad, 2),
                  "edge_width": round(edge_width, 3),
                  "edge_width_px": round(edge_width, 3)},
    )


def _estimate_edge_width(grad: np.ndarray) -> float:
    """Estimate average edge width from horizontal gradient profile spread.

    Projects gradient onto x-axis to find how spread out vertical edge
    energy is. Uses the standard deviation of gradient-weighted column
    positions as a measure of edge blur.
    """
    # Project gradient onto x-axis (sum over rows)
    gx_proj = grad.sum(axis=0)
    if gx_proj.sum() < 1e-6:
        return 10.0
    gx_proj = gx_proj / gx_proj.sum()  # normalize to probability distribution

    cols = np.arange(len(gx_proj))
    center = np.sum(cols * gx_proj)
    spread = np.sqrt(np.sum((cols - center) ** 2 * gx_proj))
    # spread ~ 0.5 pixels for a perfect step edge, larger for blurred
    return float(spread)
