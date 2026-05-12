import numpy as np
from .image_model import Image


def read_nv12(path: str, width: int, height: int) -> Image:
    """Read NV12 YUV 4:2:0 semi-planar file.

    NV12 layout:
      Y plane:  width x height bytes
      UV plane: width x height/2 bytes (interleaved U,V,U,V,...)

    Chroma is nearest-neighbor upsampled to full resolution.
    """
    raw = np.fromfile(path, dtype=np.uint8)
    y_size = width * height

    y = raw[:y_size].reshape(height, width).astype(np.float32) / 255.0

    uv = raw[y_size:y_size + width * height // 2].reshape(height // 2, width // 2, 2)
    u_half = uv[:, :, 0].astype(np.float32) / 255.0 - 0.5
    v_half = uv[:, :, 1].astype(np.float32) / 255.0 - 0.5

    cb = u_half.repeat(2, axis=0).repeat(2, axis=1)
    cr = v_half.repeat(2, axis=0).repeat(2, axis=1)

    return Image(y=y, cb=cb, cr=cr, width=width, height=height, format="nv12")
