"""Dynamic range: 2-function fusion — global DR + 9-grid regional DR mean."""

import numpy as np
from ..preprocess.pipeline import PreprocessedImage
from ..metrics import MetricResult, register


@register
def dynamic_range(pp: PreprocessedImage) -> MetricResult:
    """Score dynamic range via 2-function fusion.

    Sub-functions:
      1. Global P1-P99 DR (60%) — overall histogram spread
      2. 9-grid regional DR mean (40%) — detects local under/overexposure
    """
    y = pp.y
    is_uniform = float(y.std()) < 0.01

    # ── Sub 1: Global DR (60%) ──
    if is_uniform:
        sub_global = 70.0
        dr_stops = 0.0
    else:
        y_valid = y[(y > 0.005) & (y < 0.995)]
        if y_valid.size > 100:
            p01 = float(np.percentile(y_valid, 1))
            p99 = float(np.percentile(y_valid, 99))
            dr_stops = float(np.log2(p99 / max(p01, 0.001)))
        else:
            dr_stops = 0.0
        sub_global = max(0.0, min(100.0, (dr_stops - 2.0) / 7.0 * 100.0))

    highlight_clip = float((y > 0.98).mean())
    shadow_clip = float((y < 0.02).mean())
    clip_penalty = (highlight_clip + shadow_clip) * 200.0

    # ── Sub 2: 9-grid regional DR mean (40%) ──
    sub_regional = _regional_dr_score(y, is_uniform)

    # ── Fusion ──
    if is_uniform:
        score = 70.0
    else:
        score = 0.6 * sub_global + 0.4 * sub_regional - clip_penalty
        score = max(0.0, score)

    heatmap = (1.0 - pp.gradient_mag / (pp.gradient_mag.max() + 1e-8)).astype(np.float32)

    region_scores = {}
    for name, mask in pp.roi.items():
        if is_uniform:
            region_scores[name] = 70.0
        else:
            ym = y[mask]
            yv = ym[(ym > 0.005) & (ym < 0.995)]
            if yv.size > 10:
                p1, p9 = np.percentile(yv, 1), np.percentile(yv, 99)
                dr = float(np.log2(p9 / max(p1, 0.001))) if p1 > 0 else 0.0
                region_scores[name] = float(max(0.0, min(100.0, (dr - 2.0) / 7.0 * 100.0)))
            else:
                region_scores[name] = 0.0

    parts = []
    if is_uniform:
        parts.append("均匀画面，动态范围不适用")
    else:
        if dr_stops < 4:
            parts.append(f"全局动态范围偏低 ({dr_stops:.1f} stops)")
        if highlight_clip > 0.03:
            parts.append(f"高光裁切严重 ({highlight_clip:.1%})")
        if shadow_clip > 0.03:
            parts.append(f"暗部裁切严重 ({shadow_clip:.1%})")
        if sub_regional < 50:
            parts.append("区域间动态范围差异大，存在局部曝光问题")

    diagnosis = "；".join(parts) if parts else "动态范围正常"

    return MetricResult(
        name="dynamic_range", global_score=round(score, 1), heatmap=heatmap,
        region_scores=region_scores, diagnosis=diagnosis,
        metadata={"sub_global_dr": round(sub_global, 1),
                  "sub_regional_dr": round(sub_regional, 1),
                  "dr_stops": round(dr_stops, 2) if not is_uniform else 0.0},
    )


def _regional_dr_score(y: np.ndarray, is_uniform: bool) -> float:
    """Compute dynamic range in a 3x3 grid and return mean score."""
    if is_uniform:
        return 70.0
    h, w = y.shape
    dr_values = []
    for ri in range(3):
        for ci in range(3):
            r0, r1 = ri * h // 3, (ri + 1) * h // 3
            c0, c1 = ci * w // 3, (ci + 1) * w // 3
            block = y[r0:r1, c0:c1]
            bv = block[(block > 0.005) & (block < 0.995)]
            if bv.size > 20:
                p1 = float(np.percentile(bv, 2))
                p9 = float(np.percentile(bv, 98))
                if p1 > 0:
                    dr_values.append(float(np.log2(p9 / p1)))
    if not dr_values:
        return 0.0
    mean_dr = float(np.mean(dr_values))
    return max(0.0, min(100.0, (mean_dr - 2.0) / 7.0 * 100.0))
