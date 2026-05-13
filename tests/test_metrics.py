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


def test_get_all_metrics_returns_13():
    """Registry returns 13 metric functions."""
    metrics = get_all_metrics()
    assert len(metrics) == 13
    names = [m.__name__ for m in metrics]
    assert "exposure" in names
    assert "sharpness" in names
    assert "noise" in names


from scorer.preprocess.pipeline import PreprocessedImage
from scorer.preprocess.roi import generate_roi_masks
from scorer.metrics.exposure import exposure
from scorer.metrics.brightness import brightness
from scorer.metrics.contrast import contrast
from scorer.metrics.sharpness import sharpness
from scorer.metrics.noise import noise


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
    """Uniform image is not penalized for lacking contrast (no content defect)."""
    y = np.full((64, 64), 0.5, dtype=np.float32)
    pp = make_preprocessed(y_array=y)
    result = contrast(pp)
    assert result.global_score > 70  # uniform = no contrast defect
    assert "均匀" in result.diagnosis


def test_contrast_high_contrast():
    """Image with black and white halves gets higher contrast score."""
    y = np.zeros((64, 64), dtype=np.float32)
    y[:, 32:] = 1.0
    pp = make_preprocessed(y_array=y)
    result = contrast(pp)
    assert result.global_score > 40


def test_sharpness_blurry_image():
    """Blurred image gets lower sharpness score than sharp edges."""
    # Create an edge image, blur it
    y_sharp = np.zeros((64, 64), dtype=np.float32)
    y_sharp[:, 32:] = 1.0  # sharp vertical edge
    pp_sharp = make_preprocessed(y_array=y_sharp)
    result_sharp = sharpness(pp_sharp)

    # Blur the edge
    import cv2
    y_blurry = cv2.GaussianBlur(y_sharp, (5, 5), 2.0)
    pp_blurry = make_preprocessed(y_array=y_blurry)
    result_blurry = sharpness(pp_blurry)

    assert result_sharp.global_score > result_blurry.global_score
    assert "edge_width_px" in result_sharp.metadata


def test_noise_uniform_image():
    """Clean uniform image gets high noise score."""
    y = np.full((64, 64), 0.5, dtype=np.float32)
    cb = np.zeros((64, 64), dtype=np.float32)
    cr = np.zeros((64, 64), dtype=np.float32)
    pp = make_preprocessed(y_array=y, cb_array=cb, cr_array=cr)
    result = noise(pp)
    assert result.global_score > 70
    assert "snr_db" in result.metadata


def test_noise_noisy_image():
    """Noisy image gets lower noise score."""
    np.random.seed(0)
    y = np.full((64, 64), 0.5, dtype=np.float32) + np.random.randn(64, 64).astype(np.float32) * 0.05
    cb = np.random.randn(64, 64).astype(np.float32) * 0.02
    cr = np.random.randn(64, 64).astype(np.float32) * 0.02
    pp = make_preprocessed(y_array=y, cb_array=cb, cr_array=cr)
    result_clean = noise(pp)

    y_noisy = y + np.random.randn(64, 64).astype(np.float32) * 0.10
    pp_noisy = make_preprocessed(y_array=y_noisy, cb_array=cb, cr_array=cr)
    result_noisy = noise(pp_noisy)

    assert result_clean.global_score > result_noisy.global_score


from scorer.metrics.color_accuracy import color_accuracy
from scorer.metrics.white_balance import white_balance
from scorer.metrics.dynamic_range import dynamic_range
from scorer.metrics.texture import texture_preservation
from scorer.metrics.uniformity import uniformity
from scorer.metrics.fringing import fringing
from scorer.metrics.saturation import saturation
from scorer.metrics.distortion import distortion


def test_color_accuracy_neutral():
    """Neutral image (Cb=Cr=0) gets high color accuracy score."""
    pp = make_preprocessed(cb_array=np.zeros((64, 64), dtype=np.float32),
                           cr_array=np.zeros((64, 64), dtype=np.float32))
    result = color_accuracy(pp)
    assert result.global_score > 60
    assert "delta_c" in result.metadata


def test_white_balance_gray_world():
    """Image satisfying gray-world assumption gets high WB score."""
    pp = make_preprocessed(cb_array=np.zeros((64, 64), dtype=np.float32),
                           cr_array=np.zeros((64, 64), dtype=np.float32))
    result = white_balance(pp)
    assert result.global_score > 60


def test_dynamic_range():
    """Full-range image gets reasonable DR score."""
    y = np.linspace(0.02, 0.98, 64 * 64).reshape(64, 64).astype(np.float32)
    pp = make_preprocessed(y_array=y)
    result = dynamic_range(pp)
    assert result.global_score > 20
    assert "dr_stops" in result.metadata


def test_texture_preservation():
    """Image with detail zone gets texture score."""
    y = np.full((64, 64), 0.5, dtype=np.float32)
    y[16:48, 16:48] += np.random.randn(32, 32).astype(np.float32) * 0.02
    pp = make_preprocessed(y_array=y)
    result = texture_preservation(pp)
    assert 0 <= result.global_score <= 100


def test_uniformity_perfect():
    """Uniform image gets high uniformity score."""
    pp = make_preprocessed()
    result = uniformity(pp)
    assert result.global_score > 70
    assert "worst_corner_ratio" in result.metadata


def test_fringing_none():
    """Image with no edges has no color fringing."""
    pp = make_preprocessed()
    result = fringing(pp)
    assert result.global_score > 90  # no edges = no fringing


def test_saturation_neutral():
    """Neutral saturation (moderate chroma) gets high score."""
    cb = np.full((64, 64), 0.08, dtype=np.float32)
    cr = np.full((64, 64), 0.06, dtype=np.float32)
    pp = make_preprocessed(cb_array=cb, cr_array=cr)
    result = saturation(pp)
    assert result.global_score > 85
    assert "mean_chroma" in result.metadata


def test_saturation_desaturated():
    """Very low chroma image gets low saturation score."""
    cb = np.zeros((64, 64), dtype=np.float32)
    cr = np.zeros((64, 64), dtype=np.float32)
    pp = make_preprocessed(cb_array=cb, cr_array=cr)
    result = saturation(pp)
    assert result.global_score < 50
    assert "欠饱和" in result.diagnosis


def test_saturation_oversaturated():
    """Very high chroma image gets low saturation score."""
    cb = np.full((64, 64), 0.20, dtype=np.float32)
    cr = np.full((64, 64), 0.20, dtype=np.float32)
    pp = make_preprocessed(cb_array=cb, cr_array=cr)
    result = saturation(pp)
    assert result.global_score < 70
    assert "过饱和" in result.diagnosis


def test_distortion_flat_image():
    """Uniform image has too few edges — distortion skipped gracefully."""
    y = np.full((64, 64), 0.5, dtype=np.float32)
    pp = make_preprocessed(y_array=y)
    result = distortion(pp)
    assert result.global_score == 80.0
    assert "直线特征不足" in result.diagnosis
    assert "contours_analyzed" in result.metadata


def test_distortion_straight_edges():
    """Image with straight edges gets high distortion score."""
    import cv2
    y = np.zeros((128, 128), dtype=np.float32)
    y[40:44, 20:108] = 0.9  # horizontal line
    y[20:108, 60:64] = 0.9  # vertical line
    pp = make_preprocessed(y_array=y)
    # Refresh gradient and edge mask
    from scorer.preprocess.feature_maps import compute_gradient_mag, compute_edge_mask
    grad = compute_gradient_mag(y)
    edge = compute_edge_mask(y)
    pp.gradient_mag = grad
    pp.edge_mask = edge
    result = distortion(pp)
    assert result.global_score > 60  # straight lines = low distortion
    assert "contours_analyzed" in result.metadata
