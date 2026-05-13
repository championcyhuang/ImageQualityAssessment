from ..metrics import MetricResult
from ..aggregate import compute_total_score, flag_issues

# Algorithm descriptions for each metric
_ALGO_DESCRIPTIONS = {
    "exposure":            "双函数融合: 均值距离+裁切惩罚(60%) + 熵权曝光(40%，高熵内容区加权)",
    "brightness":          "ITU-R BT.709感知亮度，Gamma 2.2近似，目标感知亮度~0.72(18%灰)，偏离扣分",
    "contrast":            "局部RMS对比度(8x8块std/mean) + 全局Michelson对比度(p95-p5)/(p95+p5)，各占50%",
    "color_accuracy":      "灰世界假设 — Cb/Cr色度通道偏离中性灰(0,0)的程度，delta_C = sqrt(Cb^2+Cr^2)",
    "white_balance":       "三函数融合: 灰世界(45%)+白点(20%)+边缘色度一致性(35%)，估计光源色温",
    "sharpness":           "三函数融合: Laplacian方差(40%)+边缘宽度(30%)+梯度幅值均值(30%)",
    "noise":               "三函数融合: 平坦区SNR(40%)+小波多尺度噪声(35%)+局部标准差中位数(25%)",
    "dynamic_range":       "双函数融合: 全局P1-P99 DR(60%)+9宫格区域DR均值(40%)，惩罚裁切比例",
    "texture_preservation": "双函数融合: 方差比(50%)+梯度双峰分离度(50%)，检测降噪纹理抹除",
    "uniformity":          "双函数融合: 四角中心亮度比(50%)+2D二次曲面拟合残差(50%)，检测非对称shading",
    "fringing":            "高对比度边缘邻域紫边/青边检测，Cr>0.05&Cb<-0.03(紫边)/Cr<-0.03&Cb>0.03(青边)像素占比",
    "saturation":          "YCbCr色度幅值分析，chroma_mag = sqrt(Cb^2+Cr^2)，目标范围0.05-0.15，惩罚欠饱和和过饱和",
    "distortion":          "双函数融合: 轮廓直线度偏离(60%)+Hough角度集中度投票(40%)，检测径向畸变",
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
