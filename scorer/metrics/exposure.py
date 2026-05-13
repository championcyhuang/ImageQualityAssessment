"""Exposure: 2-function fusion — mean-based + entropy-weighted."""

import numpy as np
from ..preprocess.pipeline import PreprocessedImage
from ..metrics import MetricResult, register


@register
def exposure(pp: PreprocessedImage) -> MetricResult:
    """Score exposure via 2-function fusion.

    Sub-functions:
      1. Mean-Y distance + clipping penalty (60%) — basic histogram
      2. Entropy-weighted exposure (40%) — weights high-entropy (content-rich)
         regions more heavily, mimicking how AE algorithms prioritize detail
    """
    y = pp.y

    # ── Sub 1: Mean-based (60%) ──
    mean_y = float(y.mean())
    highlight_clip = float((y > 0.95).mean())
    shadow_clip = float((y < 0.05).mean())
    sub_mean = max(0.0, 100.0 - abs(mean_y - 0.5) * 200.0)
    clip_penalty = (highlight_clip + shadow_clip) * 100.0
    sub_mean_penalized = max(0.0, sub_mean - clip_penalty)

    # ── Sub 2: Entropy-weighted (40%) ──
    sub_entropy = _entropy_weighted_exposure(y)

    # ── Fusion ──
    score = 0.6 * sub_mean_penalized + 0.4 * sub_entropy

    heatmap = np.abs(y - 0.45).astype(np.float32)
    region_scores = {}
    for name, mask in pp.roi.items():
        region_scores[name] = float(max(0.0, 100.0 - abs(y[mask].mean() - 0.5) * 200.0))

    parts = []
    if highlight_clip > 0.02:
        parts.append(f"高光裁切 {highlight_clip:.1%}")
    if shadow_clip > 0.02:
        parts.append(f"暗部裁切 {shadow_clip:.1%}")
    if mean_y < 0.3:
        parts.append("整体欠曝")
    elif mean_y > 0.7:
        parts.append("整体过曝")
    if sub_entropy < 50:
        parts.append("内容区曝光不足（熵权法检测）")

    diagnosis = "；".join(parts) if parts else "曝光正常"

    return MetricResult(
        name="exposure", global_score=round(score, 1), heatmap=heatmap,
        region_scores=region_scores, diagnosis=diagnosis,
        metadata={"mean_y": round(mean_y, 4),
                  "sub_mean": round(sub_mean_penalized, 1),
                  "sub_entropy": round(sub_entropy, 1),
                  "highlight_clip": round(highlight_clip, 4),
                  "shadow_clip": round(shadow_clip, 4)},
    )


def _entropy_weighted_exposure(y: np.ndarray, block_size: int = 32) -> float:
    """Entropy-weighted exposure score.

    Divides the image into blocks. Blocks with higher entropy (more content)
    contribute more to the exposure score. This prevents large uniform areas
    (sky, wall) from dominating the mean-based score.
    """
    h, w = y.shape
    if h < block_size or w < block_size:
        return 50.0  # image too small for block analysis

    scores = []
    weights = []
    n_bins = 16

    for i in range(0, h - block_size, block_size // 2):
        for j in range(0, w - block_size, block_size // 2):
            block = y[i:i + block_size, j:j + block_size]
            hist, _ = np.histogram(block, bins=n_bins, range=(0, 1))
            hist = hist.astype(np.float32) / block.size

            # Entropy of the block
            eps = 1e-8
            entropy = -np.sum(hist * np.log2(hist + eps))
            max_entropy = np.log2(n_bins)
            entropy_norm = entropy / max_entropy  # 0-1

            # Block exposure score
            block_mean = float(block.mean())
            block_score = max(0.0, 100.0 - abs(block_mean - 0.5) * 200.0)

            scores.append(block_score)
            weights.append(entropy_norm)

    scores_arr = np.array(scores)
    weights_arr = np.array(weights)
    if weights_arr.sum() > 1e-8:
        return float(np.average(scores_arr, weights=weights_arr))
    return 50.0
