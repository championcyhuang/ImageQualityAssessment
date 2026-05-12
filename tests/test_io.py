import numpy as np
from scorer.io.image_model import Image


def test_image_dataclass_creation():
    y = np.ones((64, 64), dtype=np.float32) * 0.5
    cb = np.zeros((64, 64), dtype=np.float32)
    cr = np.zeros((64, 64), dtype=np.float32)
    img = Image(y=y, cb=cb, cr=cr, width=64, height=64, format="nv12")
    assert img.width == 64
    assert img.height == 64
    assert img.format == "nv12"
    assert img.y.shape == (64, 64)
    assert img.cb.shape == (64, 64)


import tempfile
import os
from scorer.io.nv12 import read_nv12


def test_nv12_reader_shape():
    """Write synthetic NV12, read back, check shapes."""
    w, h = 64, 64
    y_data = np.full((h, w), 128, dtype=np.uint8)
    uv_data = np.full((h // 2, w // 2, 2), 128, dtype=np.uint8)
    nv12_bytes = np.concatenate([y_data.flatten(), uv_data.flatten()])

    with tempfile.NamedTemporaryFile(suffix=".nv12", delete=False) as f:
        f.write(nv12_bytes.tobytes())
        tmp_path = f.name

    try:
        img = read_nv12(tmp_path, w, h)
        assert img.width == w
        assert img.height == h
        assert img.format == "nv12"
        assert img.y.shape == (h, w)
        assert img.cb.shape == (h, w)
        assert img.cr.shape == (h, w)
    finally:
        os.unlink(tmp_path)


def test_nv12_reader_values():
    """Verify NV12 Y plane values survive roundtrip."""
    w, h = 32, 32
    y_data = np.arange(w * h, dtype=np.uint8).reshape(h, w)
    uv_data = np.zeros((h // 2, w // 2, 2), dtype=np.uint8)
    nv12_bytes = np.concatenate([y_data.flatten(), uv_data.flatten()])

    with tempfile.NamedTemporaryFile(suffix=".nv12", delete=False) as f:
        f.write(nv12_bytes.tobytes())
        tmp_path = f.name

    try:
        img = read_nv12(tmp_path, w, h)
        expected_y = y_data.astype(np.float32) / 255.0
        assert np.allclose(img.y, expected_y, atol=0.01)
    finally:
        os.unlink(tmp_path)


from scorer.io.png_reader import read_png


def create_test_png(path, rgb_array):
    """Helper: write a uint8 RGB array as PNG via PIL."""
    from PIL import Image as PILImage
    img = PILImage.fromarray((rgb_array * 255).astype(np.uint8))
    img.save(path)


def test_png_reader_grayscale():
    """PNG reader converts grayscale to Y channel."""
    w, h = 32, 32
    gray = np.full((h, w, 3), 0.5, dtype=np.float32)  # uniform gray RGB

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    create_test_png(tmp_path, gray)

    try:
        img = read_png(tmp_path)
        assert img.width == w
        assert img.height == h
        assert img.format == "png"
        assert img.y.shape == (h, w)
        # Gray image should have Y ≈ 0.5, Cb/Cr ≈ 0
        assert np.abs(img.y.mean() - 0.5) < 0.1
    finally:
        os.unlink(tmp_path)


def test_png_reader_color():
    """PNG reader produces nonzero chroma for colored image."""
    w, h = 16, 16
    red = np.zeros((h, w, 3), dtype=np.float32)
    red[:, :, 0] = 1.0  # pure red

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    create_test_png(tmp_path, red)

    try:
        img = read_png(tmp_path)
        # Red should have positive Cr, near-zero Cb
        assert img.cr.mean() > 0.1
    finally:
        os.unlink(tmp_path)
