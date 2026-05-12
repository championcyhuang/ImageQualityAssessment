import numpy as np
from ..preprocess.pipeline import PreprocessedImage
from ..metrics import MetricResult, register


@register
def color_accuracy(pp: PreprocessedImage) -> MetricResult:
    """Score color accuracy via chroma channel deviation from gray-world neutral."""
    cb, cr = pp.cb, pp.cr
    delta_cb = float(np.mean(np.abs(cb)))
    delta_cr = float(np.mean(np.abs(cr)))
    delta_c = np.sqrt(delta_cb ** 2 + delta_cr ** 2)
    score = max(0.0, 100.0 - delta_c * 800.0)
    heatmap = (np.abs(cb) + np.abs(cr)).astype(np.float32)
    region_scores = {}
    for name, mask in pp.roi.items():
        dc = float(np.sqrt(np.mean(cb[mask]) ** 2 + np.mean(cr[mask]) ** 2))
        region_scores[name] = float(max(0.0, 100.0 - dc * 800.0))
    parts = []
    if delta_cb > 0.03:
        direction = "偏蓝" if delta_cb > 0 else "偏黄"
        parts.append(f"Cb通道偏移 ({direction}, ΔCb={delta_cb:.3f})")
    if delta_cr > 0.03:
        direction = "偏红" if delta_cr > 0 else "偏青"
        parts.append(f"Cr通道偏移 ({direction}, ΔCr={delta_cr:.3f})")
    diagnosis = "；".join(parts) if parts else "色彩还原正常"
    return MetricResult(
        name="color_accuracy", global_score=round(score, 1), heatmap=heatmap,
        region_scores=region_scores, diagnosis=diagnosis,
        metadata={"delta_cb": round(delta_cb, 4), "delta_cr": round(delta_cr, 4),
                  "delta_c": round(delta_c, 4)},
    )
