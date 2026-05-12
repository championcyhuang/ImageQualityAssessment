import numpy as np
from ..preprocess.pipeline import PreprocessedImage
from ..metrics import MetricResult, register


@register
def noise(pp: PreprocessedImage) -> MetricResult:
    """Score noise using SNR estimation and ISO 15739 visual noise approach.

    Luma noise measured on Y channel in flat regions. Chroma noise on Cb/Cr.
    Higher SNR = higher score.
    """
    y = pp.y
    cb = pp.cb
    cr = pp.cr

    # Use flat regions for noise estimation
    if pp.flat_mask.sum() > 100:
        y_flat = y[pp.flat_mask]
        cb_flat = cb[pp.flat_mask]
        cr_flat = cr[pp.flat_mask]
    else:
        y_flat = y.ravel()
        cb_flat = cb.ravel()
        cr_flat = cr.ravel()

    # Luma noise (standard deviation in flat regions)
    luma_noise = float(np.std(y_flat))

    # Chroma noise
    chroma_noise = float(np.std(cb_flat) + np.std(cr_flat))

    # SNR = signal / noise (signal ≈ mean of flat regions)
    signal = float(np.mean(y_flat))
    if luma_noise > 0:
        snr_db = float(20 * np.log10(signal / luma_noise))
    else:
        snr_db = 60.0  # essentially noiseless

    # Score: SNR > 40dB is excellent, < 20dB is poor
    snr_score = max(0.0, min(100.0, (snr_db - 15.0) / 30.0 * 100.0))
    chroma_penalty = min(40.0, chroma_noise * 200.0)
    score = max(0.0, snr_score - chroma_penalty)

    # Heatmap: local std in sliding windows
    from scipy.ndimage import uniform_filter
    h, w = y.shape
    local_mean = uniform_filter(y, size=8)
    local_sq_mean = uniform_filter(y ** 2, size=8)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))
    heatmap = local_std.astype(np.float32)

    region_scores = {}
    for name, mask in pp.roi.items():
        ym = y[mask]
        if ym.size > 0 and np.std(ym) > 0:
            sig = float(np.mean(ym))
            r_snr = float(20 * np.log10(sig / np.std(ym)))
            region_scores[name] = float(max(0.0, min(100.0, (r_snr - 15.0) / 30.0 * 100.0)))
        else:
            region_scores[name] = 100.0

    parts = []
    if luma_noise > 0.02:
        parts.append(f"亮度噪声偏高 (std={luma_noise:.4f})")
    if chroma_noise > 0.015:
        parts.append(f"色度噪声偏高 (chroma_std={chroma_noise:.4f})")
    if snr_db < 25:
        parts.append(f"信噪比偏低 ({snr_db:.1f}dB)")

    diagnosis = "；".join(parts) if parts else "噪声控制良好"

    return MetricResult(
        name="noise",
        global_score=round(score, 1),
        heatmap=heatmap,
        region_scores=region_scores,
        diagnosis=diagnosis,
        metadata={"snr_db": round(snr_db, 2), "luma_noise_std": round(luma_noise, 4),
                  "chroma_noise_std": round(chroma_noise, 4)},
    )
