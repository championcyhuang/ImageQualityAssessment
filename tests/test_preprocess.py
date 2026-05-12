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
