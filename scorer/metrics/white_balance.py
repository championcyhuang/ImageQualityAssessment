"""White balance: 3-function fusion — gray-world + white-patch + edge-based."""

import numpy as np
from ..preprocess.pipeline import PreprocessedImage
from ..metrics import MetricResult, register


@register
def white_balance(pp: PreprocessedImage) -> MetricResult:
    """Score white balance via 3-function fusion.

    Sub-functions:
      1. Gray-world (45%) — mean chroma should be neutral
      2. White-patch (20%) — brightest pixels should be neutral
      3. Edge-based constancy (35%) — chroma change consistency across edges
         (Van de Weijer 2007 edge-based color constancy)
    """
    y, cb, cr = pp.y, pp.cb, pp.cr

    # ── Sub 1: Gray-world (45%) ──
    cb_mean = float(cb.mean())
    cr_mean = float(cr.mean())
    gw_offset = np.sqrt(cb_mean ** 2 + cr_mean ** 2)
    sub_gw = max(0.0, 100.0 - gw_offset * 500.0)

    # ── Sub 2: White-patch (20%) ──
    threshold = np.percentile(y, 95)
    bright_mask = y >= threshold
    if bright_mask.sum() > 10:
        wp_cb = float(cb[bright_mask].mean())
        wp_cr = float(cr[bright_mask].mean())
        wp_offset = np.sqrt(wp_cb ** 2 + wp_cr ** 2)
    else:
        wp_offset = 0.0
    sub_wp = max(0.0, 100.0 - wp_offset * 500.0)

    # ── Sub 3: Edge-based constancy (35%) ──
    sub_edge = _edge_chroma_consistency(y, cb, cr, pp.gradient_mag)

    # ── Fusion ──
    score = 0.45 * sub_gw + 0.20 * sub_wp + 0.35 * sub_edge

    # Estimated color temperature
    if abs(cr_mean) > 1e-6:
        estimated_k = 5000.0 + cb_mean / (cr_mean + 1e-6) * 3000.0
    else:
        estimated_k = 6500.0

    heatmap = np.sqrt(cb ** 2 + cr ** 2).astype(np.float32)
    region_scores = {}
    for name, mask in pp.roi.items():
        cm = float(np.sqrt(cb[mask].mean() ** 2 + cr[mask].mean() ** 2))
        region_scores[name] = float(max(0.0, 100.0 - cm * 500.0))

    parts = []
    if gw_offset > 0.02:
        parts.append(f"灰世界偏离 (offset={gw_offset:.3f})")
    if wp_offset > 0.02:
        parts.append(f"白点不中性 (offset={wp_offset:.3f})")
    if sub_edge < 50:
        parts.append(f"边缘色度一致性差 ({sub_edge:.0f})")

    diagnosis = "；".join(parts) if parts else "白平衡正常"

    return MetricResult(
        name="white_balance", global_score=round(score, 1), heatmap=heatmap,
        region_scores=region_scores, diagnosis=diagnosis,
        metadata={"estimated_illuminant_k": round(estimated_k, 0),
                  "sub_gray_world": round(sub_gw, 1),
                  "sub_white_patch": round(sub_wp, 1),
                  "sub_edge": round(sub_edge, 1)},
    )


def _edge_chroma_consistency(y: np.ndarray, cb: np.ndarray, cr: np.ndarray,
                              gradient_mag: np.ndarray) -> float:
    """Edge-based color constancy (simplified Van de Weijer 2007).

    At strong edges, the chroma change direction should be consistent
    across the image if white balance is correct. We sample edge pixels
    and measure the variance of the chroma gradient direction.
    """
    # Find strong edge pixels (top 10% gradient)
    g_flat = gradient_mag.ravel()
    threshold = np.percentile(g_flat, 90)
    edge_pixels = gradient_mag > threshold

    if edge_pixels.sum() < 100:
        return 70.0

    # Compute chroma gradient at edge pixels
    gy_cb, gx_cb = np.gradient(cb)
    gy_cr, gx_cr = np.gradient(cr)

    # Chroma gradient direction: angle of (dCb, dCr) vector
    gcb_x = gx_cb[edge_pixels]
    gcb_y = gy_cb[edge_pixels]
    gcr_x = gx_cr[edge_pixels]
    gcr_y = gy_cr[edge_pixels]

    # Magnitude of chroma gradient
    mag = np.sqrt(gcb_x ** 2 + gcb_y ** 2 + gcr_x ** 2 + gcr_y ** 2)
    mag = np.maximum(mag, 1e-8)

    # Normalize to direction vectors
    vx = (gcb_x + gcr_x) / mag  # combined chroma change x
    vy = (gcb_y + gcr_y) / mag  # combined chroma change y

    # Consistency: circular variance of direction vectors
    # High variance = inconsistent edge chroma behavior = possible WB issue
    mean_vx = float(np.mean(vx))
    mean_vy = float(np.mean(vy))
    mean_mag = np.sqrt(mean_vx ** 2 + mean_vy ** 2)

    # mean_mag ~ 1.0 = all edges agree on chroma direction (good WB)
    # mean_mag ~ 0.0 = random directions (WB issue)
    return max(0.0, min(100.0, mean_mag * 100.0))
