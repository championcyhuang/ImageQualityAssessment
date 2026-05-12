import numpy as np
import cv2


def compute_gradient_mag(y: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude on Y channel."""
    y_u8 = (np.clip(y, 0.0, 1.0) * 255).astype(np.uint8)
    gx = cv2.Sobel(y_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y_u8, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)


def compute_edge_mask(y: np.ndarray) -> np.ndarray:
    """Canny edge detection on Y channel."""
    y_u8 = (np.clip(y, 0.0, 1.0) * 255).astype(np.uint8)
    edges = cv2.Canny(y_u8, 50, 150)
    return edges.astype(bool)


def classify_texture(y: np.ndarray, window_size: int = 8, threshold: float = 0.0005
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Classify pixels as detail (high local variance) or flat (low variance).

    Returns (detail_mask, flat_mask), both boolean.
    """
    h, w = y.shape
    detail = np.zeros((h, w), dtype=bool)
    flat = np.zeros((h, w), dtype=bool)

    for i in range(0, h - window_size + 1, window_size):
        for j in range(0, w - window_size + 1, window_size):
            block = y[i:i + window_size, j:j + window_size]
            var = block.var()
            if var > threshold:
                detail[i:i + window_size, j:j + window_size] = True
            else:
                flat[i:i + window_size, j:j + window_size] = True

    return detail, flat
