import numpy as np
from ..preprocess.pipeline import PreprocessedImage
from ..metrics import MetricResult, register


@register
def texture_preservation(pp: PreprocessedImage) -> MetricResult:
    """Score texture preservation: variance ratio in detail vs flat zones."""
    y = pp.y
    detail_var = float(np.var(y[pp.detail_mask])) if pp.detail_mask.sum() > 10 else 0.0
    flat_var = float(np.var(y[pp.flat_mask])) if pp.flat_mask.sum() > 10 else 1e-6
    if flat_var > 0:
        ratio = detail_var / flat_var
    else:
        ratio = detail_var / 1e-6 if detail_var > 0 else 1.0
    score = max(0.0, min(100.0, (ratio - 1.0) / 8.0 * 100.0))
    heatmap = np.zeros_like(y, dtype=np.float32)
    heatmap[pp.detail_mask] = 0.0
    heatmap[pp.flat_mask] = 1.0
    region_scores = {}
    for name, mask in pp.roi.items():
        detail_in_roi = mask & pp.detail_mask
        flat_in_roi = mask & pp.flat_mask
        dv = float(np.var(y[detail_in_roi])) if detail_in_roi.sum() > 10 else 0.0
        fv = float(np.var(y[flat_in_roi])) if flat_in_roi.sum() > 10 else 1e-6
        r = dv / fv if fv > 0 else 1.0
        region_scores[name] = float(max(0.0, min(100.0, (r - 1.0) / 8.0 * 100.0)))
    parts = []
    if ratio < 2.0:
        parts.append(f"纹理/平坦方差比偏低 ({ratio:.2f})，细节分辨力不足")
    diagnosis = "；".join(parts) if parts else "纹理保留正常"
    return MetricResult(
        name="texture_preservation", global_score=round(score, 1), heatmap=heatmap,
        region_scores=region_scores, diagnosis=diagnosis,
        metadata={"texture_to_flat_variance_ratio": round(ratio, 2)},
    )
