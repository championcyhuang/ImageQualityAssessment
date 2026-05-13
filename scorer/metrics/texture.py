"""Texture: 2-function fusion — variance ratio + gradient bimodal ratio."""

import numpy as np
from ..preprocess.pipeline import PreprocessedImage
from ..metrics import MetricResult, register


@register
def texture_preservation(pp: PreprocessedImage) -> MetricResult:
    """Score texture preservation via 2-function fusion.

    Sub-functions:
      1. Detail/flat variance ratio (50%) — detects texture smearing
      2. Gradient bimodal ratio (50%) — separates texture edges from noise
    """
    y = pp.y
    is_uniform = float(y.std()) < 0.01

    # ── Sub 1: Variance ratio (50%) ──
    detail_var = float(np.var(y[pp.detail_mask])) if pp.detail_mask.sum() > 10 else 0.0
    flat_var = float(np.var(y[pp.flat_mask])) if pp.flat_mask.sum() > 10 else 1e-6

    if is_uniform:
        sub_var = 80.0
    else:
        ratio = detail_var / flat_var if flat_var > 0 else 1.0
        sub_var = max(0.0, min(100.0, (ratio - 1.0) / 8.0 * 100.0))

    # ── Sub 2: Gradient bimodal ratio (50%) ──
    sub_grad = _gradient_bimodal_score(y, pp.detail_mask, pp.flat_mask, is_uniform)

    # ── Fusion ──
    if is_uniform:
        score = 80.0
    else:
        score = 0.5 * sub_var + 0.5 * sub_grad

    # Heatmap: binary detail/flat map scaled by score
    heatmap = np.zeros_like(y, dtype=np.float32)
    heatmap[pp.detail_mask] = max(0, (100 - score)) / 200.0
    heatmap[pp.flat_mask] = (100 - score) / 100.0

    region_scores = {}
    for name, mask in pp.roi.items():
        if is_uniform:
            region_scores[name] = 80.0
        else:
            di = mask & pp.detail_mask
            fi = mask & pp.flat_mask
            dv = float(np.var(y[di])) if di.sum() > 10 else 0.0
            fv = float(np.var(y[fi])) if fi.sum() > 10 else 1e-6
            r = dv / fv if fv > 0 else 1.0
            region_scores[name] = float(max(0.0, min(100.0, (r - 1.0) / 8.0 * 100.0)))

    parts = []
    if is_uniform:
        parts.append("均匀画面，纹理不适用")
    else:
        ratio_v = detail_var / flat_var if flat_var > 0 else 0.0
        if ratio_v < 2.0:
            parts.append(f"纹理/平坦方差比偏低 ({ratio_v:.2f})，细节分辨力不足")
        if sub_grad < 50:
            parts.append("梯度双峰分离度不足，纹理可能被降噪抹除")

    diagnosis = "；".join(parts) if parts else "纹理保留正常"

    return MetricResult(
        name="texture_preservation", global_score=round(score, 1), heatmap=heatmap,
        region_scores=region_scores, diagnosis=diagnosis,
        metadata={"sub_variance_ratio": round(sub_var, 1),
                  "sub_gradient_bimodal": round(sub_grad, 1)},
    )


def _gradient_bimodal_score(y: np.ndarray, detail_mask: np.ndarray,
                             flat_mask: np.ndarray, is_uniform: bool) -> float:
    """Score based on separation between detail and flat gradient distributions.

    A well-textured image has two distinct gradient populations:
    - Detail zones: high gradient (edges + texture)
    - Flat zones: low gradient (only noise)

    We measure the bimodal separation: how distinct are these two populations?
    If texture is smeared, the two distributions merge → low score.
    """
    if is_uniform:
        return 80.0

    # Compute gradient magnitude
    import cv2
    y_u8 = (np.clip(y, 0.0, 1.0) * 255).astype(np.uint8)
    gx = cv2.Sobel(y_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y_u8, cv2.CV_32F, 0, 1, ksize=3)
    gm = np.sqrt(gx ** 2 + gy ** 2)

    # Sample gradients from each zone
    g_detail = gm[detail_mask] if detail_mask.sum() > 50 else gm.ravel()
    g_flat = gm[flat_mask] if flat_mask.sum() > 50 else gm.ravel()

    if len(g_detail) < 10 or len(g_flat) < 10:
        return 50.0

    # Compute means and combined std
    mu_d = float(np.mean(g_detail))
    mu_f = float(np.mean(g_flat))
    std_pooled = float(np.sqrt((np.var(g_detail) + np.var(g_flat)) / 2))

    if std_pooled < 1e-8:
        return 50.0

    # Separation score: Cohen's d-like measure
    separation = abs(mu_d - mu_f) / std_pooled
    # separation > 2.0 = clearly bimodal (good texture), < 0.5 = merged (smeared)
    return max(0.0, min(100.0, separation / 2.5 * 100.0))
