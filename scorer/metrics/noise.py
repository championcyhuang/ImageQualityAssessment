"""Noise: 3-function fusion — flat SNR + wavelet multi-scale + local median."""

import numpy as np
from scipy.ndimage import uniform_filter
from ..preprocess.pipeline import PreprocessedImage
from ..metrics import MetricResult, register


@register
def noise(pp: PreprocessedImage) -> MetricResult:
    """Score noise via 3-function weighted fusion.

    Sub-functions:
      1. Flat-region SNR (40%) — ISO 15739 approach, luma + chroma
      2. Wavelet multi-scale noise (35%) — separates texture from noise
      3. Local std median (25%) — robust to flat-region scarcity
    """
    y, cb, cr = pp.y, pp.cb, pp.cr

    # ── Sub-function 1: Flat-region SNR (40%) ──
    if pp.flat_mask.sum() > 100:
        y_flat, cb_flat, cr_flat = y[pp.flat_mask], cb[pp.flat_mask], cr[pp.flat_mask]
    else:
        y_flat, cb_flat, cr_flat = y.ravel(), cb.ravel(), cr.ravel()

    luma_std = float(np.std(y_flat))
    chroma_std = float(np.std(cb_flat) + np.std(cr_flat))
    signal = float(np.mean(y_flat))
    snr_db = float(20 * np.log10(signal / luma_std)) if luma_std > 0 else 60.0
    sub_snr = max(0.0, min(100.0, (snr_db - 10.0) / 35.0 * 100.0))

    # ── Sub-function 2: Wavelet multi-scale noise (35%) ──
    sub_wavelet = _wavelet_noise_score(y)

    # ── Sub-function 3: Local std median (25%) ──
    h, w = y.shape
    local_mean = uniform_filter(y, size=8)
    local_sq_mean = uniform_filter(y ** 2, size=8)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))
    median_local_std = float(np.median(local_std))
    sub_local = max(0.0, min(100.0, 100.0 - median_local_std * 250.0))

    # ── Fusion ──
    chroma_penalty = min(35.0, chroma_std * 180.0)
    score = 0.4 * sub_snr + 0.35 * sub_wavelet + 0.25 * sub_local - chroma_penalty
    score = max(0.0, score)

    # Heatmap: local std
    heatmap = local_std.astype(np.float32)

    # Region scores
    region_scores = {}
    for name, mask in pp.roi.items():
        ym = y[mask]
        if ym.size > 0 and np.std(ym) > 0:
            sig = float(np.mean(ym))
            rs = float(20 * np.log10(sig / np.std(ym)))
            region_scores[name] = float(max(0.0, min(100.0, (rs - 10.0) / 35.0 * 100.0)))
        else:
            region_scores[name] = 100.0

    # Diagnosis
    parts = []
    if luma_std > 0.02:
        parts.append(f"亮度噪声偏高 (std={luma_std:.4f})")
    if chroma_std > 0.015:
        parts.append(f"色度噪声偏高 (chroma_std={chroma_std:.4f})")
    if snr_db < 20:
        parts.append(f"信噪比偏低 ({snr_db:.1f}dB)")
    if sub_wavelet < 40:
        parts.append(f"小波噪声偏高 (wavelet={sub_wavelet:.0f})")

    diagnosis = "；".join(parts) if parts else "噪声控制良好"

    return MetricResult(
        name="noise",
        global_score=round(score, 1),
        heatmap=heatmap,
        region_scores=region_scores,
        diagnosis=diagnosis,
        metadata={"snr_db": round(snr_db, 2), "luma_noise_std": round(luma_std, 4),
                  "chroma_noise_std": round(chroma_std, 4),
                  "sub_snr": round(sub_snr, 1), "sub_wavelet": round(sub_wavelet, 1),
                  "sub_local": round(sub_local, 1)},
    )


def _wavelet_noise_score(y: np.ndarray) -> float:
    """Estimate noise via simple 2-level Haar wavelet.

    Decomposes Y into approximation + detail coefficients.
    Noise manifests in the finest detail level (HH1).
    The median absolute deviation (MAD) of HH1 is a robust noise estimator.
    """
    h, w = y.shape
    # Pad to even dimensions
    hp, wp = h if h % 2 == 0 else h - 1, w if w % 2 == 0 else w - 1
    if hp < 4 or wp < 4:
        return 50.0
    yc = y[:hp, :wp]

    # Level 1 Haar decomposition
    # Approximation (LL): average of 2x2 blocks
    ll = (yc[0::2, 0::2] + yc[0::2, 1::2] + yc[1::2, 0::2] + yc[1::2, 1::2]) * 0.25
    # Horizontal detail (LH)
    lh = ((yc[0::2, 0::2] - yc[0::2, 1::2]) + (yc[1::2, 0::2] - yc[1::2, 1::2])) * 0.25
    # Vertical detail (HL)
    hl = ((yc[0::2, 0::2] - yc[1::2, 0::2]) + (yc[0::2, 1::2] - yc[1::2, 1::2])) * 0.25
    # Diagonal detail (HH) — finest noise
    hh = ((yc[0::2, 0::2] - yc[0::2, 1::2]) - (yc[1::2, 0::2] - yc[1::2, 1::2])) * 0.25

    # MAD of HH coefficients (robust noise estimate)
    hh_flat = hh.ravel()
    mad = float(np.median(np.abs(hh_flat - np.median(hh_flat))))
    # Normalize: mad < 0.003 = clean, mad > 0.03 = noisy
    noise_level = mad / 0.003
    return max(0.0, min(100.0, 100.0 - noise_level * 15.0))
