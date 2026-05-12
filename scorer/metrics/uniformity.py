import numpy as np
from ..preprocess.pipeline import PreprocessedImage
from ..metrics import MetricResult, register


@register
def uniformity(pp: PreprocessedImage) -> MetricResult:
    """Score uniformity (lens shading) via corner-to-center luminance ratio."""
    y = pp.y
    center_mask = pp.roi["center"]
    center_y = float(y[center_mask].mean()) if center_mask.sum() > 0 else 0.5
    corner_names = ["corner_tl", "corner_tr", "corner_bl", "corner_br"]
    ratios = []
    for cn in corner_names:
        mask = pp.roi[cn]
        if mask.sum() > 0 and center_y > 0:
            ratios.append(float(y[mask].mean()) / center_y)
    if ratios:
        mean_ratio = float(np.mean(ratios))
        worst_ratio = float(np.min(ratios))
    else:
        mean_ratio = 1.0
        worst_ratio = 1.0
    score = max(0.0, min(100.0, (worst_ratio - 0.5) / 0.45 * 100.0))
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
        worst_cn = corner_names[np.argmin(ratios)] if ratios else "unknown"
        parts.append(f"最严重区域: {worst_cn}")
    if mean_ratio < 0.85:
        parts.append(f"整体均匀性不足 (平均比={mean_ratio:.2f})")
    diagnosis = "；".join(parts) if parts else "画面均匀性良好"
    return MetricResult(
        name="uniformity", global_score=round(score, 1), heatmap=heatmap,
        region_scores=region_scores, diagnosis=diagnosis,
        metadata={"corner_vs_center_luma_ratio": round(worst_ratio, 3),
                  "mean_corner_ratio": round(mean_ratio, 3)},
    )
