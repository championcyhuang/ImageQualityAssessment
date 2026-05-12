import numpy as np
from ..preprocess.pipeline import PreprocessedImage
from ..metrics import MetricResult, register


@register
def brightness(pp: PreprocessedImage) -> MetricResult:
    """Score perceived brightness using ITU-R BT.709 weighting.

    Target perceived brightness around 0.45-0.55 (50-60 in 100 scale).
    Diagnoses over/under brightness per ROI.
    """
    y = pp.y

    # BT.709 perceived brightness weighting (nonlinear gamma approximation)
    perceived = np.power(y, 1.0 / 2.2)
    mean_perceived = float(perceived.mean())

    # Ideal perceived brightness approx 0.72 (18% gray in sRGB after gamma)
    target = 0.72
    score = max(0.0, 100.0 - abs(mean_perceived - target) * 150.0)

    heatmap = np.abs(perceived - target).astype(np.float32)

    region_scores = {}
    for name, mask in pp.roi.items():
        pm = perceived[mask].mean()
        region_scores[name] = float(max(0.0, 100.0 - abs(pm - target) * 150.0))

    parts = []
    if mean_perceived < target - 0.15:
        parts.append(f"整体偏暗 (perceived={mean_perceived:.3f})")
    elif mean_perceived > target + 0.15:
        parts.append(f"整体偏亮 (perceived={mean_perceived:.3f})")

    diagnosis = "；".join(parts) if parts else "亮度适中"

    return MetricResult(
        name="brightness",
        global_score=round(score, 1),
        heatmap=heatmap,
        region_scores=region_scores,
        diagnosis=diagnosis,
        metadata={"perceived_brightness": round(mean_perceived, 4)},
    )
