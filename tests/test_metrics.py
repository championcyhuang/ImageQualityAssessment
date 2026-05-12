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


from scorer.preprocess.pipeline import PreprocessedImage
from scorer.preprocess.roi import generate_roi_masks
from scorer.metrics.exposure import exposure
from scorer.metrics.brightness import brightness
from scorer.metrics.contrast import contrast


def make_preprocessed(y_array=None, cb_array=None, cr_array=None):
    """Helper to create a PreprocessedImage for testing."""
    h, w = y_array.shape if y_array is not None else (64, 64)
    y = y_array if y_array is not None else np.full((h, w), 0.5, dtype=np.float32)
    cb = cb_array if cb_array is not None else np.zeros((h, w), dtype=np.float32)
    cr = cr_array if cr_array is not None else np.zeros((h, w), dtype=np.float32)
    roi = generate_roi_masks(w, h)
    detail = np.zeros((h, w), dtype=bool)
    flat = np.ones((h, w), dtype=bool)
    edge = np.zeros((h, w), dtype=bool)
    grad = np.zeros((h, w), dtype=np.float32)
    return PreprocessedImage(y=y, cb=cb, cr=cr, width=w, height=h,
                             roi=roi, detail_mask=detail, flat_mask=flat,
                             edge_mask=edge, gradient_mag=grad, format="test")


def test_exposure_perfect_midgray():
    """Mid-gray image (Y=0.5) gets high exposure score."""
    y = np.full((64, 64), 0.5, dtype=np.float32)
    pp = make_preprocessed(y_array=y)
    result = exposure(pp)
    assert result.global_score > 70
    assert "mean_y" in result.metadata


def test_exposure_underexposed():
    """Very dark image gets low exposure score."""
    y = np.full((64, 64), 0.05, dtype=np.float32)
    pp = make_preprocessed(y_array=y)
    result = exposure(pp)
    assert result.global_score < 50
    assert "shadow" in result.diagnosis.lower() or "暗" in result.diagnosis or result.global_score < 60


def test_brightness_midgray():
    """Mid-gray has target perceived brightness, score near 80."""
    y = np.full((64, 64), 0.5, dtype=np.float32)
    pp = make_preprocessed(y_array=y)
    result = brightness(pp)
    assert result.global_score > 60
    assert "perceived_brightness" in result.metadata


def test_contrast_flat_image():
    """Uniform image has low contrast score."""
    y = np.full((64, 64), 0.5, dtype=np.float32)
    pp = make_preprocessed(y_array=y)
    result = contrast(pp)
    assert result.global_score < 50  # flat = low contrast


def test_contrast_high_contrast():
    """Image with black and white halves gets higher contrast score."""
    y = np.zeros((64, 64), dtype=np.float32)
    y[:, 32:] = 1.0
    pp = make_preprocessed(y_array=y)
    result = contrast(pp)
    assert result.global_score > 40
