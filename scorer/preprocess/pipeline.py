from dataclasses import dataclass
import numpy as np
from ..io.image_model import Image
from .roi import generate_roi_masks
from .feature_maps import compute_gradient_mag, compute_edge_mask, classify_texture


@dataclass
class PreprocessedImage:
    y: np.ndarray
    cb: np.ndarray
    cr: np.ndarray
    width: int
    height: int
    roi: dict[str, np.ndarray]
    detail_mask: np.ndarray
    flat_mask: np.ndarray
    edge_mask: np.ndarray
    gradient_mag: np.ndarray
    format: str = "unknown"


def preprocess(img: Image) -> PreprocessedImage:
    """Run full preprocessing pipeline on an Image."""
    roi = generate_roi_masks(img.width, img.height)
    detail_mask, flat_mask = classify_texture(img.y)
    edge_mask = compute_edge_mask(img.y)
    gradient_mag = compute_gradient_mag(img.y)
    return PreprocessedImage(
        y=img.y, cb=img.cb, cr=img.cr,
        width=img.width, height=img.height,
        roi=roi, detail_mask=detail_mask, flat_mask=flat_mask,
        edge_mask=edge_mask, gradient_mag=gradient_mag,
        format=img.format,
    )
