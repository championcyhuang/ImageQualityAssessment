import numpy as np
from ..preprocess.pipeline import PreprocessedImage
from ..metrics import MetricResult, register


@register
def white_balance(pp: PreprocessedImage) -> MetricResult:
    """Score white balance using gray-world + white-patch assumptions."""
    y, cb, cr = pp.y, pp.cb, pp.cr
    cb_mean = float(cb.mean())
    cr_mean = float(cr.mean())
    gw_offset = np.sqrt(cb_mean ** 2 + cr_mean ** 2)
    threshold = np.percentile(y, 95)
    bright_mask = y >= threshold
    if bright_mask.sum() > 10:
        wp_cb = float(cb[bright_mask].mean())
        wp_cr = float(cr[bright_mask].mean())
        wp_offset = np.sqrt(wp_cb ** 2 + wp_cr ** 2)
    else:
        wp_offset = 0.0
    total_offset = 0.7 * gw_offset + 0.3 * wp_offset
    score = max(0.0, 100.0 - total_offset * 500.0)
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
    diagnosis = "；".join(parts) if parts else "白平衡正常"
    return MetricResult(
        name="white_balance", global_score=round(score, 1), heatmap=heatmap,
        region_scores=region_scores, diagnosis=diagnosis,
        metadata={"estimated_illuminant_k": round(estimated_k, 0),
                  "gray_world_offset": round(gw_offset, 4),
                  "white_patch_offset": round(wp_offset, 4)},
    )
