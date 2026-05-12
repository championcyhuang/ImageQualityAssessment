from ..metrics import MetricResult
from ..aggregate import compute_total_score, flag_issues


def print_report(results: list[MetricResult], total_score: float | None = None):
    """Print formatted quality report to console."""
    if total_score is None:
        total_score = compute_total_score(results)

    print("=" * 70)
    print(f"  图像质量打分报告 (总分: {total_score:.1f}/100)")
    print("=" * 70)
    print(f"{'指标':<16} {'分数':>6} {'状态':<10}")
    print("-" * 70)

    issues = flag_issues(results)
    for r in results:
        status = "⚠️ 待优化" if r.global_score < 60 else "✓ 正常"
        print(f"  {r.name:<14} {r.global_score:>6.1f}  {status:<10}")

    print("=" * 70)

    if issues:
        print("\n🔍 问题诊断:")
        for r in issues:
            print(f"  [{r.name}] 得分 {r.global_score:.1f}")
            print(f"    → {r.diagnosis}")
        print()

    print("💡 调试建议:")
    _print_recommendations(results)


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
