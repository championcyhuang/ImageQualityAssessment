import numpy as np
from ..preprocess.pipeline import PreprocessedImage
from ..metrics import MetricResult, register


@register
def fringing(pp: PreprocessedImage) -> MetricResult:
    """Detect color fringing (purple/cyan) near high-contrast edges."""
    cb, cr = pp.cb, pp.cr
    edge = pp.edge_mask
    if edge.sum() < 10:
        return MetricResult(
            name="fringing", global_score=100.0,
            heatmap=np.zeros_like(cb, dtype=np.float32),
            region_scores={name: 100.0 for name in pp.roi},
            diagnosis="无明显紫边（边缘像素不足）",
            metadata={"fringe_pixel_ratio": 0.0},
        )
    from scipy.ndimage import binary_dilation
    edge_zone = binary_dilation(edge, iterations=2)
    purple = (cr > 0.05) & (cb < -0.03) & edge_zone
    cyan = (cr < -0.03) & (cb > 0.03) & edge_zone
    fringe = purple | cyan
    fringe_ratio = float(fringe.sum()) / max(edge_zone.sum(), 1)
    score = max(0.0, 100.0 - fringe_ratio * 100.0 * 15.0)
    heatmap = (fringe.astype(np.float32) * (100.0 - score) / 100.0)
    region_scores = {}
    for name, mask in pp.roi.items():
        zone = edge_zone & mask
        fz = fringe & mask
        ratio = float(fz.sum()) / max(zone.sum(), 1)
        region_scores[name] = float(max(0.0, 100.0 - ratio * 100.0 * 15.0))
    parts = []
    if fringe_ratio > 0.01:
        n_purple = int(purple.sum())
        n_cyan = int(cyan.sum())
        parts.append(f"检测到色散边纹 (紫边{n_purple}px, 青边{n_cyan}px")
        parts.append(f"色散占比 {fringe_ratio:.2%}")
    worst_region = min(region_scores, key=region_scores.get) if region_scores else "unknown"
    if region_scores.get(worst_region, 100) < 80:
        parts.append(f"最严重区域: {worst_region}")
    diagnosis = "；".join(parts) if parts else "无明显紫边/色散"
    return MetricResult(
        name="fringing", global_score=round(score, 1), heatmap=heatmap,
        region_scores=region_scores, diagnosis=diagnosis,
        metadata={"fringe_pixel_ratio": round(fringe_ratio, 4)},
    )
