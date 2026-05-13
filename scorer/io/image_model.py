from dataclasses import dataclass
import numpy as np


@dataclass
class Image:
    """Raw image data in YCbCr space, float32, full resolution."""
    y: np.ndarray      # float32, shape (H, W)
    cb: np.ndarray     # float32, shape (H, W), upsampled to full res
    cr: np.ndarray     # float32, shape (H, W), upsampled to full res
    width: int
    height: int
    format: str = "unknown"  # "nv12" or "png"
    rgb: np.ndarray | None = None  # original RGB uint8 (H,W,3), for PNG sources
