import numpy as np
from ..preprocess.pipeline import PreprocessedImage
from ..metrics import MetricResult, register


@register
def contrast(pp: PreprocessedImage) -> MetricResult:
    """Score contrast using local RMS contrast and global Michelson contrast.

    RMS contrast is computed per 8x8 block, then averaged. Michelson contrast
    uses the 5th and 95th percentiles.
    """
    y = pp.y
    h, w = y.shape

    # Local RMS contrast (8x8 blocks)
    rms_values = []
    for i in range(0, h - 7, 8):
        for j in range(0, w - 7, 8):
            block = y[i:i + 8, j:j + 8]
            mu = block.mean()
            if mu > 0:
                rms = np.sqrt(np.mean((block - mu) ** 2)) / mu
                rms_values.append(rms)

    if rms_values:
        rms_contrast = float(np.mean(rms_values))
    else:
        rms_contrast = 0.0

    # Global Michelson contrast
    p5 = float(np.percentile(y, 5))
    p95 = float(np.percentile(y, 95))
    if p95 + p5 > 0:
        michelson = (p95 - p5) / (p95 + p5)
    else:
        michelson = 0.0

    # Score: RMS contrast ~0.15 is good, Michelson ~0.8 is good
    rms_score = max(0.0, min(100.0, rms_contrast / 0.20 * 100.0))
    mich_score = max(0.0, min(100.0, michelson / 0.85 * 100.0))
    score = 0.5 * rms_score + 0.5 * mich_score

    heatmap = np.abs(pp.gradient_mag).astype(np.float32)
    # Normalize heatmap
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max() * (100 - score) / 100.0

    region_scores = {}
    for name, mask in pp.roi.items():
        ym = y[mask]
        mu = ym.mean()
        if mu > 0:
            rms_r = float(np.sqrt(np.mean((ym - mu) ** 2)) / mu)
            region_scores[name] = float(max(0.0, min(100.0, rms_r / 0.20 * 100.0)))
        else:
            region_scores[name] = 0.0

    parts = []
    if rms_contrast < 0.05:
        parts.append("局部对比度偏低，画面偏平")
    if michelson < 0.3:
        parts.append("全局对比度不足")

    diagnosis = "；".join(parts) if parts else "对比度正常"

    return MetricResult(
        name="contrast",
        global_score=round(score, 1),
        heatmap=heatmap,
        region_scores=region_scores,
        diagnosis=diagnosis,
        metadata={"rms_contrast": round(rms_contrast, 4),
                  "michelson_contrast": round(michelson, 4)},
    )
