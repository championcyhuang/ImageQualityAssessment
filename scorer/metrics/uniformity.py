"""Uniformity: 2-function fusion — corner ratio + 2D quadratic fit residual."""

import numpy as np
from ..preprocess.pipeline import PreprocessedImage
from ..metrics import MetricResult, register


@register
def uniformity(pp: PreprocessedImage) -> MetricResult:
    """Score illumination uniformity via 2-function fusion.

    Sub-functions:
      1. Corner-to-center luminance ratio (50%) — captures extreme shading
      2. 2D quadratic fit residual (50%) — detects asymmetric shading patterns
         and non-radial falloff that corner ratios miss
    """
    y = pp.y
    h, w = y.shape
    center_mask = pp.roi["center"]
    center_y = float(y[center_mask].mean()) if center_mask.sum() > 0 else 0.5

    # ── Sub 1: Corner ratio (50%) ──
    corner_names = ["corner_tl", "corner_tr", "corner_bl", "corner_br"]
    ratios = []
    for cn in corner_names:
        mask = pp.roi[cn]
        if mask.sum() > 0 and center_y > 0:
            ratios.append(float(y[mask].mean()) / center_y)
    worst_ratio = float(np.min(ratios)) if ratios else 1.0
    mean_ratio = float(np.mean(ratios)) if ratios else 1.0
    sub_corner = max(0.0, min(100.0, (worst_ratio - 0.5) / 0.45 * 100.0))

    # ── Sub 2: 2D quadratic fit residual (50%) ──
    sub_quad = _quadratic_fit_uniformity(y, center_y)

    # ── Fusion ──
    score = 0.5 * sub_corner + 0.5 * sub_quad

    heatmap = np.abs(y - center_y).astype(np.float32)
    region_scores = {}
    for name, mask in pp.roi.items():
        if mask.sum() > 0 and center_y > 0:
            r = float(y[mask].mean()) / center_y
            region_scores[name] = float(max(0.0, min(100.0, (r - 0.5) / 0.45 * 100.0)))
        else:
            region_scores[name] = 100.0

    parts = []
    if worst_ratio < 0.80:
        parts.append(f"角落亮度衰减明显 (最差角落/中心比={worst_ratio:.2f})")
        parts.append(f"最严重区域: {corner_names[np.argmin(ratios)]}")
    if mean_ratio < 0.85:
        parts.append(f"整体均匀性不足 (平均比={mean_ratio:.2f})")
    if sub_quad < 50:
        parts.append("检测到非对称 shading 模式")

    diagnosis = "；".join(parts) if parts else "画面均匀性良好"

    return MetricResult(
        name="uniformity", global_score=round(score, 1), heatmap=heatmap,
        region_scores=region_scores, diagnosis=diagnosis,
        metadata={"sub_corner_ratio": round(sub_corner, 1),
                  "sub_quad_fit": round(sub_quad, 1),
                  "worst_corner_ratio": round(worst_ratio, 3)},
    )


def _quadratic_fit_uniformity(y: np.ndarray, center_y: float) -> float:
    """Score uniformity by fitting a 2D quadratic surface.

    Model: z(x,y) = a*x^2 + b*y^2 + c*xy + d*x + e*y + f
    The residual std after fitting indicates deviation from a smooth shading
    pattern. A perfectly uniform image has zero residual.
    An image with only radial shading fits the quadratic well (low residual).
    Asymmetric or patchy shading produces high residual.
    """
    h, w = y.shape
    if h < 8 or w < 8:
        return 70.0

    # Build coordinate grid (normalized to [-1, 1])
    xs = np.linspace(-1, 1, w)
    ys = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(xs, ys)

    # Design matrix for 2nd-order polynomial: [x^2, y^2, xy, x, y, 1]
    A = np.column_stack([
        xx.ravel() ** 2, yy.ravel() ** 2, (xx * yy).ravel(),
        xx.ravel(), yy.ravel(), np.ones_like(xx.ravel())
    ])
    z = y.ravel()

    try:
        coeffs, residuals, rank, sv = np.linalg.lstsq(A, z, rcond=None)
    except np.linalg.LinAlgError:
        return 70.0

    # Reconstruct fitted surface
    z_fit = A @ coeffs
    z_fit = z_fit.reshape(h, w)

    # Residual: deviation from smooth quadratic surface
    residual = y - z_fit
    residual_rms = float(np.sqrt(np.mean(residual ** 2)))

    # Also check: how much of the original shading is explained by the fit?
    # Residual RMS / total RMS → fraction unexplained
    total_rms = float(np.sqrt(np.mean((y - center_y) ** 2)))
    if total_rms > 1e-8:
        unexplained = residual_rms / total_rms
    else:
        unexplained = 0.0

    # Low unexplained = smooth shading (quadratic fits well) = good
    # High unexplained = irregular shading (possible defect)
    return max(0.0, min(100.0, 100.0 - unexplained * 100.0))
