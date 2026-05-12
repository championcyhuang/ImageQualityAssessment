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
