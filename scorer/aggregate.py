from .metrics import MetricResult


DEFAULT_WEIGHTS = {
    "exposure": 1.0,
    "brightness": 0.8,
    "contrast": 1.0,
    "color_accuracy": 1.2,
    "white_balance": 1.0,
    "sharpness": 1.5,
    "noise": 1.2,
    "dynamic_range": 1.0,
    "texture_preservation": 1.0,
    "uniformity": 0.8,
    "fringing": 0.5,
    "saturation": 0.8,
    "distortion": 0.8,
}


def compute_total_score(results: list[MetricResult],
                        weights: dict[str, float] | None = None) -> float:
    """Weighted total score from individual metric results."""
    if weights is None:
        weights = DEFAULT_WEIGHTS
    total_weight = 0.0
    weighted_sum = 0.0
    for r in results:
        w = weights.get(r.name, 1.0)
        weighted_sum += r.global_score * w
        total_weight += w
    return round(weighted_sum / total_weight, 1) if total_weight > 0 else 0.0


def compute_deltas(results_test: list[MetricResult],
                   results_ref: list[MetricResult]) -> list[dict]:
    """Per-metric delta between test and reference device."""
    ref_map = {r.name: r for r in results_ref}
    deltas = []
    for r in results_test:
        ref = ref_map.get(r.name)
        if ref:
            deltas.append({
                "metric": r.name,
                "test_score": r.global_score,
                "ref_score": ref.global_score,
                "delta": round(r.global_score - ref.global_score, 1),
            })
    return deltas


def flag_issues(results: list[MetricResult], threshold: float = 60.0) -> list[MetricResult]:
    """Return metrics that scored below threshold."""
    return [r for r in results if r.global_score < threshold]
