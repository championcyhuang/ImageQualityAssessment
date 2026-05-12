import numpy as np
from PIL import Image as PILImage
from .image_model import Image


def read_png(path: str) -> Image:
    """Read PNG as RGB, convert to YCbCr (BT.709)."""
    pil_img = PILImage.open(path).convert("RGB")
    rgb = np.array(pil_img, dtype=np.float32) / 255.0
    h, w = rgb.shape[:2]
    y, cb, cr = _rgb_to_ycbcr_bt709(rgb)
    return Image(y=y, cb=cb, cr=cr, width=w, height=h, format="png")


def _rgb_to_ycbcr_bt709(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BT.709 full-range RGB -> YCbCr."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    y  = 0.2126 * r + 0.7152 * g + 0.0722 * b
    cb = (b - y) / 1.8556
    cr = (r - y) / 1.5748
    return y.astype(np.float32), cb.astype(np.float32), cr.astype(np.float32)
