import numpy as np
from ..preprocess.pipeline import PreprocessedImage
from ..metrics import MetricResult, register


@register
def dynamic_range(pp: PreprocessedImage) -> MetricResult:
    """Score dynamic range from histogram spread and clipping."""
    y = pp.y
    y_valid = y[(y > 0.005) & (y < 0.995)]

    # Uniform image guard: no spread means no dynamic range defect
    is_uniform = float(y.std()) < 0.01
    if is_uniform:
        score = 70.0
    else:
        if y_valid.size > 100:
            p01 = float(np.percentile(y_valid, 1))
            p99 = float(np.percentile(y_valid, 99))
            if p01 > 0:
                dr_stops = float(np.log2(p99 / p01))
            else:
                dr_stops = float(np.log2(p99 / 0.001))
        else:
            dr_stops = 0.0
        highlight_clip = float((y > 0.98).mean())
        shadow_clip = float((y < 0.02).mean())
        headroom_stops = float(-np.log2(max(highlight_clip, 0.001)))
        score = max(0.0, min(100.0, (dr_stops - 3.0) / 6.0 * 100.0))
        clip_penalty = (highlight_clip + shadow_clip) * 200.0
        score = max(0.0, score - clip_penalty)
    heatmap = (1.0 - pp.gradient_mag / (pp.gradient_mag.max() + 1e-8)).astype(np.float32)
    region_scores = {}
    for name, mask in pp.roi.items():
        if is_uniform:
            region_scores[name] = 70.0
        else:
            ym = y[mask]
            yv = ym[(ym > 0.005) & (ym < 0.995)]
            if yv.size > 10 and np.percentile(yv, 1) > 0:
                dr = float(np.log2(np.percentile(yv, 99) / np.percentile(yv, 1)))
                region_scores[name] = float(max(0.0, min(100.0, (dr - 3.0) / 6.0 * 100.0)))
            else:
                region_scores[name] = 0.0
    parts = []
    if is_uniform:
        parts.append("均匀画面，动态范围不适用")
    elif dr_stops < 5:
        parts.append(f"动态范围偏低 ({dr_stops:.1f} stops)")
    if not is_uniform and highlight_clip > 0.03:
        parts.append(f"高光裁切严重 ({highlight_clip:.1%})")
    if not is_uniform and shadow_clip > 0.03:
        parts.append(f"暗部裁切严重 ({shadow_clip:.1%})")
    diagnosis = "；".join(parts) if parts else "动态范围正常"
    return MetricResult(
        name="dynamic_range", global_score=round(score, 1), heatmap=heatmap,
        region_scores=region_scores, diagnosis=diagnosis,
        metadata={"dr_stops": round(dr_stops, 2) if not is_uniform else 0.0,
                  "headroom_stops": round(headroom_stops, 2) if not is_uniform else 0.0},
    )
