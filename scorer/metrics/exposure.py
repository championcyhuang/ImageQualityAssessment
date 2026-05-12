import numpy as np
from ..preprocess.pipeline import PreprocessedImage
from ..metrics import MetricResult, register


@register
def exposure(pp: PreprocessedImage) -> MetricResult:
    """Score exposure based on histogram analysis of Y channel.

    Target: mean Y between 0.35 and 0.65. Penalizes highlight clipping
    (Y > 0.95) and shadow clipping (Y < 0.05).
    """
    y = pp.y
    mean_y = float(y.mean())
    highlight_clip = float((y > 0.95).mean())
    shadow_clip = float((y < 0.05).mean())

    # Score based on distance from ideal mean (0.5)
    mean_score = max(0.0, 100.0 - abs(mean_y - 0.5) * 200.0)
    # Penalize clipping
    clip_penalty = (highlight_clip + shadow_clip) * 100.0
    score = max(0.0, mean_score - clip_penalty)

    # Heatmap: per-pixel brightness distance from ideal
    heatmap = np.abs(y - 0.45).astype(np.float32)

    # Per-region scores
    region_scores = {}
    for name, mask in pp.roi.items():
        region_scores[name] = float(max(0.0, 100.0 - abs(y[mask].mean() - 0.5) * 200.0))

    # Diagnosis
    parts = []
    if highlight_clip > 0.02:
        parts.append(f"高光裁切 {highlight_clip:.1%}")
    if shadow_clip > 0.02:
        parts.append(f"暗部裁切 {shadow_clip:.1%}")
    if mean_y < 0.3:
        parts.append("整体欠曝")
    elif mean_y > 0.7:
        parts.append("整体过曝")

    diagnosis = "；".join(parts) if parts else "曝光正常"

    return MetricResult(
        name="exposure",
        global_score=round(score, 1),
        heatmap=heatmap,
        region_scores=region_scores,
        diagnosis=diagnosis,
        metadata={"mean_y": round(mean_y, 4), "highlight_clip": round(highlight_clip, 4),
                  "shadow_clip": round(shadow_clip, 4)},
    )
