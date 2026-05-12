import numpy as np
from scorer.metrics import MetricResult, get_all_metrics


def test_metric_result_creation():
    """MetricResult stores all required fields."""
    hm = np.zeros((8, 8), dtype=np.float32)
    result = MetricResult(
        name="test_metric",
        global_score=75.0,
        heatmap=hm,
        region_scores={"center": 80.0, "edge_top": 70.0},
        diagnosis="Test diagnosis text",
        metadata={"raw_value": 0.5},
    )
    assert result.name == "test_metric"
    assert result.global_score == 75.0
    assert result.heatmap.shape == (8, 8)
    assert "center" in result.region_scores


def test_get_all_metrics_returns_11():
    """Registry returns 11 metric functions."""
    metrics = get_all_metrics()
    assert len(metrics) == 11
    names = [m.__name__ for m in metrics]
    assert "exposure" in names
    assert "sharpness" in names
    assert "noise" in names
