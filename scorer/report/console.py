from ..metrics import MetricResult
from ..aggregate import compute_total_score, flag_issues

# Algorithm descriptions for each metric
_ALGO_DESCRIPTIONS = {
    "exposure":            "Y通道直方图分析，目标均值0.35-0.65，惩罚高光裁切(Y>0.95)和暗部裁切(Y<0.05)",
    "brightness":          "ITU-R BT.709感知亮度，Gamma 2.2近似，目标感知亮度~0.72(18%灰)，偏离扣分",
    "contrast":            "局部RMS对比度(8x8块std/mean) + 全局Michelson对比度(p95-p5)/(p95+p5)，各占50%",
    "color_accuracy":      "灰世界假设 — Cb/Cr色度通道偏离中性灰(0,0)的程度，delta_C = sqrt(Cb^2+Cr^2)",
    "white_balance":       "灰世界(70%)+白点(30%)双假设法，分析最亮像素(95%百分位)的色度偏移，估算光源色温",
    "sharpness":           "Laplacian方差(经典对焦测度)，cv2.Laplacian + 梯度剖面散度估算边缘宽度(简化MTF法)",
    "noise":               "ISO 15739视觉噪声法 — 平坦区SNR估计(20*log10(signal/noise))，亮度噪声+色度噪声综合评分",
    "dynamic_range":       "直方图有效像素P1-P99范围，DR = log2(p99/p01) stops，惩罚高光/暗部裁切比例",
    "texture_preservation": "平坦区与细节区方差比值法，检测降噪是否过度抹除纹理，ratio = detail_var / flat_var",
    "uniformity":          "四角vs中心亮度比(镜头Shading评估)，角落亮度/中心亮度，取最差角落比值评分",
    "fringing":            "高对比度边缘邻域紫边/青边检测，Cr>0.05&Cb<-0.03(紫边)/Cr<-0.03&Cb>0.03(青边)像素占比",
    "saturation":          "YCbCr色度幅值分析，chroma_mag = sqrt(Cb^2+Cr^2)，目标范围0.05-0.15，惩罚欠饱和和过饱和",
    "distortion":          "轮廓直线度分析——检测长边缘、拟合直线、测量点线平均偏离，偏离越大畸变越严重",
}


def print_report(results: list[MetricResult], total_score: float | None = None):
    """Print formatted quality report to console."""
    if total_score is None:
        total_score = compute_total_score(results)

    print("=" * 70)
    print(f"  图像质量打分报告 (总分: {total_score:.1f}/100)")
    print("=" * 70)
    print(f"{'指标':<20} {'分数':>6} {'状态':<10}")
    print("-" * 70)

    issues = flag_issues(results)
    for r in results:
        status = "[!!] 待优化" if r.global_score < 60 else "[OK] 正常"
        print(f"  {r.name:<18} {r.global_score:>6.1f}  {status:<10}")

    print("=" * 70)

    if issues:
        print("\n[问题诊断:]")
        for r in issues:
            print(f"  [{r.name}] 得分 {r.global_score:.1f}")
            print(f"    → {r.diagnosis}")
        print()

    print("[调试建议:]")
    _print_recommendations(results)

    print()
    print("[算法说明:]")
    print("-" * 70)
    for r in results:
        desc = _ALGO_DESCRIPTIONS.get(r.name, "")
        print(f"  {r.name:<20} | {desc}")
    print("-" * 70)


def _print_recommendations(results: list[MetricResult]):
    score_map = {r.name: r.global_score for r in results}

    if score_map.get("sharpness", 100) < 60:
        for r in results:
            if r.name == "sharpness" and r.region_scores:
                worst = min(r.region_scores, key=r.region_scores.get)
                print(f"  清晰度: 检查 {worst} 区域, 可能需要调整镜头对焦或锐化参数")

    if score_map.get("noise", 100) < 60:
        print(f"  噪声: 考虑降低ISO, 或调整降噪强度(注意平衡纹理保留)")

    if score_map.get("white_balance", 100) < 60:
        print(f"  白平衡: 检查AWB算法色温估计, 可能存在色温偏差")

    if score_map.get("exposure", 100) < 60:
        print(f"  曝光: 检查AE target设置, 目标Y均值应接近0.45-0.55")

    if score_map.get("uniformity", 100) < 60:
        print(f"  均匀性: 检查镜头shading校正, 角落亮度不应低于中心80%")
