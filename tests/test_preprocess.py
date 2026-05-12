import numpy as np
from scorer.preprocess.color_space import rgb_to_ycbcr, ycbcr_to_rgb


def test_rgb_to_ycbcr_gray():
    """Gray RGB should have zero chroma channels."""
    rgb = np.full((8, 8, 3), 0.5, dtype=np.float32)
    y, cb, cr = rgb_to_ycbcr(rgb)
    assert y.shape == (8, 8)
    assert np.allclose(y, 0.5, atol=0.01)
    assert np.allclose(cb, 0.0, atol=0.01)
    assert np.allclose(cr, 0.0, atol=0.01)


def test_ycbcr_to_rgb_roundtrip():
    """RGB -> YCbCr -> RGB should be lossless."""
    rng = np.random.default_rng(42)
    rgb_in = rng.random((16, 16, 3)).astype(np.float32)
    y, cb, cr = rgb_to_ycbcr(rgb_in)
    rgb_out = ycbcr_to_rgb(y, cb, cr)
    assert np.allclose(rgb_in, rgb_out, atol=0.005)


def test_ycbcr_to_rgb_clips():
    """YCbCr -> RGB should clip to [0, 1]."""
    y = np.full((4, 4), 0.5, dtype=np.float32)
    cb = np.full((4, 4), 0.6, dtype=np.float32)  # extreme blue
    cr = np.full((4, 4), 0.0, dtype=np.float32)
    rgb = ycbcr_to_rgb(y, cb, cr)
    assert rgb.min() >= 0.0
    assert rgb.max() <= 1.0


from scorer.preprocess.roi import generate_roi_masks


def test_roi_masks_coverage():
    """ROI masks cover the full image and don't overlap."""
    masks = generate_roi_masks(width=100, height=100)
    assert "center" in masks
    assert "edge_top" in masks
    assert "edge_bot" in masks
    assert "edge_left" in masks
    assert "edge_right" in masks
    assert "corner_tl" in masks
    assert "corner_tr" in masks
    assert "corner_bl" in masks
    assert "corner_br" in masks
    assert len(masks) == 9


def test_roi_center_coverage():
    """Center ROI covers approximately 25% of image area."""
    masks = generate_roi_masks(width=100, height=100)
    center_fraction = masks["center"].sum() / (100 * 100)
    assert 0.20 < center_fraction < 0.30


def test_roi_no_overlap():
    """ROI masks should not overlap with each other."""
    masks = generate_roi_masks(width=100, height=100)
    cumulative = np.zeros((100, 100), dtype=bool)
    for mask in masks.values():
        assert not np.any(cumulative & mask)
        cumulative |= mask
    # The 9 masks should cover the entire image
    assert np.all(cumulative)


from scorer.preprocess.feature_maps import compute_gradient_mag, classify_texture
from scorer.io.image_model import Image


def test_gradient_mag_uniform():
    """Gradient magnitude is zero for uniform image."""
    y = np.full((32, 32), 0.5, dtype=np.float32)
    grad = compute_gradient_mag(y)
    assert np.allclose(grad, 0.0, atol=1e-4)


def test_gradient_mag_has_edges():
    """Gradient magnitude > 0 at sharp transitions."""
    y = np.zeros((32, 32), dtype=np.float32)
    y[:, 16:] = 1.0  # vertical edge
    grad = compute_gradient_mag(y)
    assert grad[:, 15:17].max() > 0.1
    assert grad[:, :10].max() < 0.01  # flat region


def test_classify_texture():
    """Texture classification: flat vs detail separation."""
    np.random.seed(42)
    flat = np.full((32, 32), 0.5, dtype=np.float32)
    detail = np.full((32, 32), 0.5, dtype=np.float32) + np.random.randn(32, 32).astype(np.float32) * 0.05
    y = np.concatenate([flat, detail], axis=1)  # left half flat, right half textured

    detail_mask, flat_mask = classify_texture(y, window_size=8, threshold=0.0005)
    # Left half should be classified flat
    assert flat_mask[:, :16].mean() > 0.5
    # Right half should be classified detail
    assert detail_mask[:, 16:].mean() > 0.3
