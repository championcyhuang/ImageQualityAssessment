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
