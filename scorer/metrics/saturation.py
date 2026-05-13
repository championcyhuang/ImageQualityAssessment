import numpy as np
from ..preprocess.pipeline import PreprocessedImage
from ..metrics import MetricResult, register


@register
def saturation(pp: PreprocessedImage) -> MetricResult:
    """Score color saturation via chroma magnitude in YCbCr space.

    Chroma magnitude = sqrt(Cb^2 + Cr^2). Target mean chroma between
    0.05 and 0.15 (natural-looking saturation). Penalizes both
    undersaturation (flat/dull) and oversaturation (garish/artificial).
    """
    cb, cr = pp.cb, pp.cr
    chroma_mag = np.sqrt(cb ** 2 + cr ** 2)
    mean_chroma = float(chroma_mag.mean())

    # Score: ideal range 0.05-0.15, center at 0.10
    if mean_chroma < 0.03:
        # Severely desaturated
        score = mean_chroma / 0.03 * 60.0
    elif mean_chroma < 0.05:
        # Mildly desaturated
        score = 60.0 + (mean_chroma - 0.03) / 0.02 * 30.0
    elif mean_chroma <= 0.15:
        # Sweet spot
        score = 90.0 - abs(mean_chroma - 0.10) / 0.05 * 20.0
    elif mean_chroma <= 0.25:
        # Mildly oversaturated
        score = 70.0 - (mean_chroma - 0.15) / 0.10 * 40.0
    else:
        # Severely oversaturated
        score = max(0.0, 30.0 - (mean_chroma - 0.25) * 100.0)

    score = max(0.0, min(100.0, score))

    # Heatmap: per-pixel chroma distance from ideal
    heatmap = np.abs(chroma_mag - 0.10).astype(np.float32)

    # Per-region scores
    region_scores = {}
    for name, mask in pp.roi.items():
        cm = float(chroma_mag[mask].mean())
        if cm < 0.03:
            region_scores[name] = float(cm / 0.03 * 60.0)
        elif cm < 0.05:
            region_scores[name] = float(60.0 + (cm - 0.03) / 0.02 * 30.0)
        elif cm <= 0.15:
            region_scores[name] = float(90.0 - abs(cm - 0.10) / 0.05 * 20.0)
        elif cm <= 0.25:
            region_scores[name] = float(70.0 - (cm - 0.15) / 0.10 * 40.0)
        else:
            region_scores[name] = float(max(0.0, 30.0 - (cm - 0.25) * 100.0))
        region_scores[name] = max(0.0, min(100.0, region_scores[name]))

    # Chroma variance: very low variance = uniformly flat color (possible desaturation)
    chroma_std = float(chroma_mag.std())

    parts = []
    if mean_chroma < 0.03:
        parts.append(f"严重欠饱和 (平均色度={mean_chroma:.4f})")
    elif mean_chroma < 0.05:
        parts.append(f"轻微欠饱和 (平均色度={mean_chroma:.4f})")
    if mean_chroma > 0.25:
        parts.append(f"严重过饱和 (平均色度={mean_chroma:.4f})")
    elif mean_chroma > 0.15:
        parts.append(f"轻微过饱和 (平均色度={mean_chroma:.4f})")
    if chroma_std < 0.01 and mean_chroma < 0.05:
        parts.append("画面色彩单一，缺乏色彩层次")

    diagnosis = "；".join(parts) if parts else "饱和度正常"

    return MetricResult(
        name="saturation",
        global_score=round(score, 1),
        heatmap=heatmap,
        region_scores=region_scores,
        diagnosis=diagnosis,
        metadata={"mean_chroma": round(mean_chroma, 4),
                  "chroma_std": round(chroma_std, 4)},
    )
