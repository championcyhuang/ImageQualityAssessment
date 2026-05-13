# Image Quality Scoring System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an 11-metric camera image quality scoring pipeline that reads .nv12/.png images, scores each quality dimension 0–100 with spatial diagnostics, and generates a console report + visual report PNG.

**Architecture:** 5-layer pipeline — Image I/O → Preprocessing (YCbCr + ROI + feature maps) → 11 parallel Metric Modules → Aggregation → Report. Each metric returns `MetricResult(name, global_score, heatmap, region_scores, diagnosis, metadata)`. CLI via `run.py`.

**Tech Stack:** numpy 2.4.4, scipy 1.17.1, opencv-python 4.13.0, scikit-image 0.26.0, matplotlib 3.10.8, pillow 12.2.0. Venv at `F:\AI\imglab\.venv`.

---

### Task 1: Project Scaffolding & Image Model

**Files:**
- Create: `F:\AI\AItuning\scorer\__init__.py`
- Create: `F:\AI\AItuning\scorer\io\__init__.py`
- Create: `F:\AI\AItuning\scorer\io\image_model.py`
- Create: `F:\AI\AItuning\scorer\preprocess\__init__.py`
- Create: `F:\AI\AItuning\scorer\metrics\__init__.py`
- Create: `F:\AI\AItuning\scorer\report\__init__.py`
- Create: `F:\AI\AItuning\tests\__init__.py`
- Create: `F:\AI\AItuning\tests\test_io.py`

- [ ] **Step 1: Create directory structure and the Image dataclass**

```bash
mkdir -p F:/AI/AItuning/scorer/io
mkdir -p F:/AI/AItuning/scorer/preprocess
mkdir -p F:/AI/AItuning/scorer/metrics
mkdir -p F:/AI/AItuning/scorer/report
mkdir -p F:/AI/AItuning/tests
```

- [ ] **Step 2: Write empty `__init__.py` files for all packages**

Create `F:\AI\AItuning\scorer\__init__.py`, `F:\AI\AItuning\scorer\io\__init__.py`, `F:\AI\AItuning\scorer\preprocess\__init__.py`, `F:\AI\AItuning\scorer\metrics\__init__.py`, `F:\AI\AItuning\scorer\report\__init__.py`, `F:\AI\AItuning\tests\__init__.py` — all empty files.

- [ ] **Step 3: Write the Image dataclass**

Write `F:\AI\AItuning\scorer\io\image_model.py`:

```python
from dataclasses import dataclass, field
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
```

- [ ] **Step 4: Write the first test — dataclass creation**

Write `F:\AI\AItuning\tests\test_io.py`:

```python
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
```

- [ ] **Step 5: Run test**

Run: `F:/AI/imglab/.venv/Scripts/python.exe -m pytest F:/AI/AItuning/tests/test_io.py::test_image_dataclass_creation -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add F:/AI/AItuning/scorer/ F:/AI/AItuning/tests/
git commit -m "feat: project scaffolding and Image dataclass"
```

---

### Task 2: NV12 Reader

**Files:**
- Create: `F:\AI\AItuning\scorer\io\nv12.py`
- Modify: `F:\AI\AItuning\tests\test_io.py`

- [ ] **Step 1: Write failing test for NV12 reader**

Append to `F:\AI\AItuning\tests\test_io.py`:

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `F:/AI/imglab/.venv/Scripts/python.exe -m pytest F:/AI/AItuning/tests/test_io.py::test_nv12_reader_shape -v`
Expected: FAIL with ImportError or ModuleNotFoundError

- [ ] **Step 3: Implement read_nv12**

Write `F:\AI\AItuning\scorer\io\nv12.py`:

```python
import numpy as np
from .image_model import Image


def read_nv12(path: str, width: int, height: int) -> Image:
    """Read NV12 YUV 4:2:0 semi-planar file.

    NV12 layout:
      Y plane:  width × height bytes
      UV plane: width × height/2 bytes (interleaved U,V,U,V,...)

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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `F:/AI/imglab/.venv/Scripts/python.exe -m pytest F:/AI/AItuning/tests/test_io.py::test_nv12_reader_shape F:/AI/AItuning/tests/test_io.py::test_nv12_reader_values -v`
Expected: both PASS

- [ ] **Step 5: Commit**

```bash
git add F:/AI/AItuning/scorer/io/nv12.py F:/AI/AItuning/tests/test_io.py
git commit -m "feat: NV12 reader with chroma upsampling"
```

---

### Task 3: PNG Reader & Format Detection

**Files:**
- Create: `F:\AI\AItuning\scorer\io\png_reader.py`
- Modify: `F:\AI\AItuning\tests\test_io.py`

- [ ] **Step 1: Write failing test for PNG reader**

Append to `F:\AI\AItuning\tests\test_io.py`:

```python
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
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `F:/AI/imglab/.venv/Scripts/python.exe -m pytest F:/AI/AItuning/tests/test_io.py::test_png_reader_grayscale -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement read_png with BT.709 RGB→YCbCr**

Write `F:\AI\AItuning\scorer\io\png_reader.py`:

```python
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
    """BT.709 full-range RGB → YCbCr."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    y  = 0.2126 * r + 0.7152 * g + 0.0722 * b
    cb = (b - y) / 1.8556
    cr = (r - y) / 1.5748
    return y.astype(np.float32), cb.astype(np.float32), cr.astype(np.float32)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `F:/AI/imglab/.venv/Scripts/python.exe -m pytest F:/AI/AItuning/tests/test_io.py::test_png_reader_grayscale F:/AI/AItuning/tests/test_io.py::test_png_reader_color -v`
Expected: both PASS

- [ ] **Step 5: Commit**

```bash
git add F:/AI/AItuning/scorer/io/png_reader.py F:/AI/AItuning/tests/test_io.py
git commit -m "feat: PNG reader with BT.709 RGB→YCbCr conversion"
```

---

### Task 4: Color Space Utilities

**Files:**
- Create: `F:\AI\AItuning\scorer\preprocess\color_space.py`
- Create: `F:\AI\AItuning\tests\test_preprocess.py`

- [ ] **Step 1: Write color space tests**

Write `F:\AI\AItuning\tests\test_preprocess.py`:

```python
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
    """RGB → YCbCr → RGB should be lossless."""
    rng = np.random.default_rng(42)
    rgb_in = rng.random((16, 16, 3)).astype(np.float32)
    y, cb, cr = rgb_to_ycbcr(rgb_in)
    rgb_out = ycbcr_to_rgb(y, cb, cr)
    assert np.allclose(rgb_in, rgb_out, atol=0.005)


def test_ycbcr_to_rgb_clips():
    """YCbCr → RGB should clip to [0, 1]."""
    y = np.full((4, 4), 0.5, dtype=np.float32)
    cb = np.full((4, 4), 0.6, dtype=np.float32)  # extreme blue
    cr = np.full((4, 4), 0.0, dtype=np.float32)
    rgb = ycbcr_to_rgb(y, cb, cr)
    assert rgb.min() >= 0.0
    assert rgb.max() <= 1.0
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `F:/AI/imglab/.venv/Scripts/python.exe -m pytest F:/AI/AItuning/tests/test_preprocess.py -v`
Expected: FAIL

- [ ] **Step 3: Implement color space conversions**

Write `F:\AI\AItuning\scorer\preprocess\color_space.py`:

```python
import numpy as np


def rgb_to_ycbcr(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BT.709 RGB → YCbCr. Input: (H, W, 3) float32 in [0, 1].

    Returns (Y, Cb, Cr) each (H, W) float32.
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    y  = 0.2126 * r + 0.7152 * g + 0.0722 * b
    cb = (b - y) / 1.8556
    cr = (r - y) / 1.5748
    return y.astype(np.float32), cb.astype(np.float32), cr.astype(np.float32)


def ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    """BT.709 YCbCr → RGB. Returns (H, W, 3) float32, clipped to [0, 1]."""
    r = y + 1.5748 * cr
    g = y - 0.1873 * cb - 0.4681 * cr
    b = y + 1.8556 * cb
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `F:/AI/imglab/.venv/Scripts/python.exe -m pytest F:/AI/AItuning/tests/test_preprocess.py -v`
Expected: all 3 PASS

- [ ] **Step 5: Commit**

```bash
git add F:/AI/AItuning/scorer/preprocess/color_space.py F:/AI/AItuning/tests/test_preprocess.py
git commit -m "feat: BT.709 RGB↔YCbCr color space conversion"
```

---

### Task 5: ROI Mask Generation

**Files:**
- Create: `F:\AI\AItuning\scorer\preprocess\roi.py`
- Modify: `F:\AI\AItuning\tests\test_preprocess.py`

- [ ] **Step 1: Write ROI mask tests**

Append to `F:\AI\AItuning\tests\test_preprocess.py`:

```python
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
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `F:/AI/imglab/.venv/Scripts/python.exe -m pytest F:/AI/AItuning/tests/test_preprocess.py::test_roi_masks_coverage -v`
Expected: FAIL

- [ ] **Step 3: Implement ROI mask generation**

Write `F:\AI\AItuning\scorer\preprocess\roi.py`:

```python
import numpy as np


def generate_roi_masks(width: int, height: int) -> dict[str, np.ndarray]:
    """Generate non-overlapping boolean masks for 9 ROI regions.

    center  — middle 50% width × 50% height (25% area)
    edges   — top/bot 25% height full-width; left/right 50% height × 25% width
    corners — four 25% width × 25% height squares
    """
    h_half = height // 2
    h_qtr = height // 4
    w_half = width // 2
    w_qtr = width // 4

    masks = {}

    # Center: middle 50% x 50%
    masks["center"] = _rect_mask(width, height, w_qtr, h_qtr, w_half, h_half)

    # Edges: strips at top, bottom, left, right
    masks["edge_top"]    = _rect_mask(width, height, 0, 0, width, h_qtr)
    masks["edge_bot"]    = _rect_mask(width, height, 0, height - h_qtr, width, h_qtr)
    masks["edge_left"]   = _rect_mask(width, height, 0, h_qtr, w_qtr, h_half)
    masks["edge_right"]  = _rect_mask(width, height, width - w_qtr, h_qtr, w_qtr, h_half)

    # Corners
    masks["corner_tl"] = _rect_mask(width, height, 0, 0, w_qtr, h_qtr)
    masks["corner_tr"] = _rect_mask(width, height, width - w_qtr, 0, w_qtr, h_qtr)
    masks["corner_bl"] = _rect_mask(width, height, 0, height - h_qtr, w_qtr, h_qtr)
    masks["corner_br"] = _rect_mask(width, height, width - w_qtr, height - h_qtr, w_qtr, h_qtr)

    return masks


def _rect_mask(w: int, h: int, x: int, y: int, rw: int, rh: int) -> np.ndarray:
    """Create boolean mask for rectangle (x, y, rw, rh) in (h, w) image."""
    mask = np.zeros((h, w), dtype=bool)
    mask[y:y + rh, x:x + rw] = True
    return mask
```

- [ ] **Step 4: Run tests to verify pass**

Run: `F:/AI/imglab/.venv/Scripts/python.exe -m pytest F:/AI/AItuning/tests/test_preprocess.py::test_roi_masks_coverage F:/AI/AItuning/tests/test_preprocess.py::test_roi_center_coverage F:/AI/AItuning/tests/test_preprocess.py::test_roi_no_overlap -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add F:/AI/AItuning/scorer/preprocess/roi.py F:/AI/AItuning/tests/test_preprocess.py
git commit -m "feat: ROI mask generation (center, edges, corners)"
```

---

### Task 6: Feature Maps & PreprocessedImage

**Files:**
- Create: `F:\AI\AItuning\scorer\preprocess\feature_maps.py`
- Create: `F:\AI\AItuning\scorer\preprocess\pipeline.py`
- Modify: `F:\AI\AItuning\tests\test_preprocess.py`

- [ ] **Step 1: Write feature maps test**

Append to `F:\AI\AItuning\tests\test_preprocess.py`:

```python
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
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `F:/AI/imglab/.venv/Scripts/python.exe -m pytest F:/AI/AItuning/tests/test_preprocess.py::test_gradient_mag_uniform -v`
Expected: FAIL

- [ ] **Step 3: Implement feature_maps**

Write `F:\AI\AItuning\scorer\preprocess\feature_maps.py`:

```python
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
```

- [ ] **Step 4: Implement PreprocessedImage and preprocessing pipeline**

Write `F:\AI\AItuning\scorer\preprocess\pipeline.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify pass**

Run: `F:/AI/imglab/.venv/Scripts/python.exe -m pytest F:/AI/AItuning/tests/test_preprocess.py::test_gradient_mag_uniform F:/AI/AItuning/tests/test_preprocess.py::test_gradient_mag_has_edges F:/AI/AItuning/tests/test_preprocess.py::test_classify_texture -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add F:/AI/AItuning/scorer/preprocess/feature_maps.py F:/AI/AItuning/scorer/preprocess/pipeline.py F:/AI/AItuning/tests/test_preprocess.py
git commit -m "feat: gradient, edge, texture maps and PreprocessedImage pipeline"
```

---

### Task 7: MetricResult & Metric Registry

**Files:**
- Modify: `F:\AI\AItuning\scorer\metrics\__init__.py`
- Create: `F:\AI\AItuning\tests\test_metrics.py`

- [ ] **Step 1: Write test for MetricResult and registry**

Write `F:\AI\AItuning\tests\test_metrics.py`:

```python
import numpy as np
from scorer.metrics import MetricResult, get_all_metrics


def test_metric_result_creation():
    """MetricResult stores all required fields."""
    hm = np.zeros((8, 8), dtype=np.float32)
    result = MetricResult(
        name="test_metric",
        global_score=75.0,
        heatmap=hm,
        region_scores={"center": 80.0, "edge_top": 70.0},
        diagnosis="Test diagnosis text",
        metadata={"raw_value": 0.5},
    )
    assert result.name == "test_metric"
    assert result.global_score == 75.0
    assert result.heatmap.shape == (8, 8)
    assert "center" in result.region_scores


def test_get_all_metrics_returns_11():
    """Registry returns 11 metric functions."""
    metrics = get_all_metrics()
    assert len(metrics) == 11
    names = [m.__name__ for m in metrics]
    assert "exposure" in names
    assert "sharpness" in names
    assert "noise" in names
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `F:/AI/imglab/.venv/Scripts/python.exe -m pytest F:/AI/AItuning/tests/test_metrics.py::test_metric_result_creation -v`
Expected: FAIL

- [ ] **Step 3: Implement MetricResult and registry**

Write `F:\AI\AItuning\scorer\metrics\__init__.py`:

```python
from dataclasses import dataclass
import numpy as np


@dataclass
class MetricResult:
    name: str
    global_score: float        # 0–100
    heatmap: np.ndarray        # (H, W) float32, pixel-level quality
    region_scores: dict[str, float]
    diagnosis: str
    metadata: dict


# Registry of all metric functions, filled by imports below
_METRIC_REGISTRY: list = []


def register(func):
    _METRIC_REGISTRY.append(func)
    return func


def get_all_metrics() -> list:
    return list(_METRIC_REGISTRY)
```

- [ ] **Step 4: Run MetricResult test to verify pass**

Run: `F:/AI/imglab/.venv/Scripts/python.exe -m pytest F:/AI/AItuning/tests/test_metrics.py::test_metric_result_creation -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add F:/AI/AItuning/scorer/metrics/__init__.py F:/AI/AItuning/tests/test_metrics.py
git commit -m "feat: MetricResult dataclass and metric registry"
```

---

### Task 8: Exposure, Brightness, Contrast Metrics

**Files:**
- Create: `F:\AI\AItuning\scorer\metrics\exposure.py`
- Create: `F:\AI\AItuning\scorer\metrics\brightness.py`
- Create: `F:\AI\AItuning\scorer\metrics\contrast.py`
- Modify: `F:\AI\AItuning\scorer\metrics\__init__.py`
- Modify: `F:\AI\AItuning\tests\test_metrics.py`

- [ ] **Step 1: Write tests for exposure, brightness, contrast**

Append to `F:\AI\AItuning\tests\test_metrics.py`:

```python
import numpy as np
from scorer.preprocess.pipeline import PreprocessedImage
from scorer.preprocess.roi import generate_roi_masks
from scorer.metrics.exposure import exposure
from scorer.metrics.brightness import brightness
from scorer.metrics.contrast import contrast


def make_preprocessed(y_array=None, cb_array=None, cr_array=None):
    """Helper to create a PreprocessedImage for testing."""
    h, w = y_array.shape if y_array is not None else (64, 64)
    y = y_array if y_array is not None else np.full((h, w), 0.5, dtype=np.float32)
    cb = cb_array if cb_array is not None else np.zeros((h, w), dtype=np.float32)
    cr = cr_array if cr_array is not None else np.zeros((h, w), dtype=np.float32)
    roi = generate_roi_masks(w, h)
    detail = np.zeros((h, w), dtype=bool)
    flat = np.ones((h, w), dtype=bool)
    edge = np.zeros((h, w), dtype=bool)
    grad = np.zeros((h, w), dtype=np.float32)
    return PreprocessedImage(y=y, cb=cb, cr=cr, width=w, height=h,
                             roi=roi, detail_mask=detail, flat_mask=flat,
                             edge_mask=edge, gradient_mag=grad, format="test")


def test_exposure_perfect_midgray():
    """Mid-gray image (Y=0.5) gets high exposure score."""
    y = np.full((64, 64), 0.5, dtype=np.float32)
    pp = make_preprocessed(y_array=y)
    result = exposure(pp)
    assert result.global_score > 70
    assert "mean_y" in result.metadata


def test_exposure_underexposed():
    """Very dark image gets low exposure score."""
    y = np.full((64, 64), 0.05, dtype=np.float32)
    pp = make_preprocessed(y_array=y)
    result = exposure(pp)
    assert result.global_score < 50
    assert "shadow" in result.diagnosis.lower() or "暗" in result.diagnosis or result.global_score < 60


def test_brightness_midgray():
    """Mid-gray has target perceived brightness, score near 80."""
    y = np.full((64, 64), 0.5, dtype=np.float32)
    pp = make_preprocessed(y_array=y)
    result = brightness(pp)
    assert result.global_score > 60
    assert "perceived_brightness" in result.metadata


def test_contrast_flat_image():
    """Uniform image has low contrast score."""
    y = np.full((64, 64), 0.5, dtype=np.float32)
    pp = make_preprocessed(y_array=y)
    result = contrast(pp)
    assert result.global_score < 50  # flat = low contrast


def test_contrast_high_contrast():
    """Image with black and white halves gets higher contrast score."""
    y = np.zeros((64, 64), dtype=np.float32)
    y[:, 32:] = 1.0
    pp = make_preprocessed(y_array=y)
    result = contrast(pp)
    assert result.global_score > 40
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `F:/AI/imglab/.venv/Scripts/python.exe -m pytest F:/AI/AItuning/tests/test_metrics.py::test_exposure_perfect_midgray -v`
Expected: FAIL

- [ ] **Step 3: Implement exposure metric**

Write `F:\AI\AItuning\scorer\metrics\exposure.py`:

```python
import numpy as np
from ..preprocess.pipeline import PreprocessedImage
from ..metrics import MetricResult, register


@register
def exposure(pp: PreprocessedImage) -> MetricResult:
    """Score exposure based on histogram analysis of Y channel.

    Target: mean Y between 0.35 and 0.65. Penalizes highlight clipping
    (Y > 0.95) and shadow clipping (Y < 0.05).
    """
    y = pp.y
    mean_y = float(y.mean())
    highlight_clip = float((y > 0.95).mean())
    shadow_clip = float((y < 0.05).mean())

    # Score based on distance from ideal mean (0.5)
    mean_score = max(0.0, 100.0 - abs(mean_y - 0.5) * 200.0)
    # Penalize clipping
    clip_penalty = (highlight_clip + shadow_clip) * 100.0
    score = max(0.0, mean_score - clip_penalty)

    # Heatmap: per-pixel brightness distance from ideal
    heatmap = np.abs(y - 0.45).astype(np.float32)

    # Per-region scores
    region_scores = {}
    for name, mask in pp.roi.items():
        region_scores[name] = float(max(0.0, 100.0 - abs(y[mask].mean() - 0.5) * 200.0))

    # Diagnosis
    parts = []
    if highlight_clip > 0.02:
        parts.append(f"高光裁切 {highlight_clip:.1%}")
    if shadow_clip > 0.02:
        parts.append(f"暗部裁切 {shadow_clip:.1%}")
    if mean_y < 0.3:
        parts.append("整体欠曝")
    elif mean_y > 0.7:
        parts.append("整体过曝")

    diagnosis = "；".join(parts) if parts else "曝光正常"

    return MetricResult(
        name="exposure",
        global_score=round(score, 1),
        heatmap=heatmap,
        region_scores=region_scores,
        diagnosis=diagnosis,
        metadata={"mean_y": round(mean_y, 4), "highlight_clip": round(highlight_clip, 4),
                  "shadow_clip": round(shadow_clip, 4)},
    )
```

- [ ] **Step 4: Implement brightness metric**

Write `F:\AI\AItuning\scorer\metrics\brightness.py`:

```python
import numpy as np
from ..preprocess.pipeline import PreprocessedImage
from ..metrics import MetricResult, register


@register
def brightness(pp: PreprocessedImage) -> MetricResult:
    """Score perceived brightness using ITU-R BT.709 weighting.

    Target perceived brightness around 0.45–0.55 (50–60 in 100 scale).
    Diagnoses over/under brightness per ROI.
    """
    y = pp.y

    # BT.709 perceived brightness weighting (nonlinear gamma approximation)
    perceived = np.power(y, 1.0 / 2.2)
    mean_perceived = float(perceived.mean())

    # Ideal perceived brightness ≈ 0.72 (18% gray in sRGB after gamma)
    target = 0.72
    score = max(0.0, 100.0 - abs(mean_perceived - target) * 150.0)

    heatmap = np.abs(perceived - target).astype(np.float32)

    region_scores = {}
    for name, mask in pp.roi.items():
        pm = perceived[mask].mean()
        region_scores[name] = float(max(0.0, 100.0 - abs(pm - target) * 150.0))

    parts = []
    if mean_perceived < target - 0.15:
        parts.append(f"整体偏暗 (perceived={mean_perceived:.3f})")
    elif mean_perceived > target + 0.15:
        parts.append(f"整体偏亮 (perceived={mean_perceived:.3f})")

    diagnosis = "；".join(parts) if parts else "亮度适中"

    return MetricResult(
        name="brightness",
        global_score=round(score, 1),
        heatmap=heatmap,
        region_scores=region_scores,
        diagnosis=diagnosis,
        metadata={"perceived_brightness": round(mean_perceived, 4)},
    )
```

- [ ] **Step 5: Implement contrast metric**

Write `F:\AI\AItuning\scorer\metrics\contrast.py`:

```python
import numpy as np
from ..preprocess.pipeline import PreprocessedImage
from ..metrics import MetricResult, register


@register
def contrast(pp: PreprocessedImage) -> MetricResult:
    """Score contrast using local RMS contrast and global Michelson contrast.

    RMS contrast is computed per 8×8 block, then averaged. Michelson contrast
    uses the 5th and 95th percentiles.
    """
    y = pp.y
    h, w = y.shape

    # Local RMS contrast (8×8 blocks)
    rms_values = []
    for i in range(0, h - 7, 8):
        for j in range(0, w - 7, 8):
            block = y[i:i + 8, j:j + 8]
            mu = block.mean()
            if mu > 0:
                rms = np.sqrt(np.mean((block - mu) ** 2)) / mu
                rms_values.append(rms)

    if rms_values:
        rms_contrast = float(np.mean(rms_values))
    else:
        rms_contrast = 0.0

    # Global Michelson contrast
    p5 = float(np.percentile(y, 5))
    p95 = float(np.percentile(y, 95))
    if p95 + p5 > 0:
        michelson = (p95 - p5) / (p95 + p5)
    else:
        michelson = 0.0

    # Score: RMS contrast ~0.15 is good, Michelson ~0.8 is good
    rms_score = max(0.0, min(100.0, rms_contrast / 0.20 * 100.0))
    mich_score = max(0.0, min(100.0, michelson / 0.85 * 100.0))
    score = 0.5 * rms_score + 0.5 * mich_score

    heatmap = np.abs(pp.gradient_mag).astype(np.float32)
    # Normalize heatmap
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max() * (100 - score) / 100.0

    region_scores = {}
    for name, mask in pp.roi.items():
        ym = y[mask]
        mu = ym.mean()
        if mu > 0:
            rms_r = float(np.sqrt(np.mean((ym - mu) ** 2)) / mu)
            region_scores[name] = float(max(0.0, min(100.0, rms_r / 0.20 * 100.0)))
        else:
            region_scores[name] = 0.0

    parts = []
    if rms_contrast < 0.05:
        parts.append("局部对比度偏低，画面偏平")
    if michelson < 0.3:
        parts.append("全局对比度不足")

    diagnosis = "；".join(parts) if parts else "对比度正常"

    return MetricResult(
        name="contrast",
        global_score=round(score, 1),
        heatmap=heatmap,
        region_scores=region_scores,
        diagnosis=diagnosis,
        metadata={"rms_contrast": round(rms_contrast, 4),
                  "michelson_contrast": round(michelson, 4)},
    )
```

- [ ] **Step 6: Run metric tests**

Run: `F:/AI/imglab/.venv/Scripts/python.exe -m pytest F:/AI/AItuning/tests/test_metrics.py::test_exposure_perfect_midgray F:/AI/AItuning/tests/test_metrics.py::test_exposure_underexposed F:/AI/AItuning/tests/test_metrics.py::test_brightness_midgray F:/AI/AItuning/tests/test_metrics.py::test_contrast_flat_image F:/AI/AItuning/tests/test_metrics.py::test_contrast_high_contrast -v`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add F:/AI/AItuning/scorer/metrics/exposure.py F:/AI/AItuning/scorer/metrics/brightness.py F:/AI/AItuning/scorer/metrics/contrast.py F:/AI/AItuning/tests/test_metrics.py F:/AI/AItuning/scorer/metrics/__init__.py
git commit -m "feat: exposure, brightness, contrast metrics"
```

---

### Task 9: Sharpness & Noise Metrics

**Files:**
- Create: `F:\AI\AItuning\scorer\metrics\sharpness.py`
- Create: `F:\AI\AItuning\scorer\metrics\noise.py`
- Modify: `F:\AI\AItuning\tests\test_metrics.py`

- [ ] **Step 1: Write sharpness and noise tests**

Append to `F:\AI\AItuning\tests\test_metrics.py`:

```python
from scorer.metrics.sharpness import sharpness
from scorer.metrics.noise import noise


def test_sharpness_blurry_image():
    """Blurred image gets lower sharpness score than sharp edges."""
    # Create an edge image, blur it
    y_sharp = np.zeros((64, 64), dtype=np.float32)
    y_sharp[:, 32:] = 1.0  # sharp vertical edge
    pp_sharp = make_preprocessed(y_array=y_sharp)
    result_sharp = sharpness(pp_sharp)

    # Blur the edge
    import cv2
    y_blurry = cv2.GaussianBlur(y_sharp, (5, 5), 2.0)
    pp_blurry = make_preprocessed(y_array=y_blurry)
    result_blurry = sharpness(pp_blurry)

    assert result_sharp.global_score > result_blurry.global_score
    assert "edge_width" in result_sharp.metadata


def test_noise_uniform_image():
    """Clean uniform image gets high noise score."""
    y = np.full((64, 64), 0.5, dtype=np.float32)
    cb = np.zeros((64, 64), dtype=np.float32)
    cr = np.zeros((64, 64), dtype=np.float32)
    pp = make_preprocessed(y_array=y, cb_array=cb, cr_array=cr)
    result = noise(pp)
    assert result.global_score > 70
    assert "snr_db" in result.metadata


def test_noise_noisy_image():
    """Noisy image gets lower noise score."""
    np.random.seed(0)
    y = np.full((64, 64), 0.5, dtype=np.float32) + np.random.randn(64, 64).astype(np.float32) * 0.05
    cb = np.random.randn(64, 64).astype(np.float32) * 0.02
    cr = np.random.randn(64, 64).astype(np.float32) * 0.02
    pp = make_preprocessed(y_array=y, cb_array=cb, cr_array=cr)
    result_clean = noise(pp)

    y_noisy = y + np.random.randn(64, 64).astype(np.float32) * 0.10
    pp_noisy = make_preprocessed(y_array=y_noisy, cb_array=cb, cr_array=cr)
    result_noisy = noise(pp_noisy)

    assert result_clean.global_score > result_noisy.global_score
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `F:/AI/imglab/.venv/Scripts/python.exe -m pytest F:/AI/AItuning/tests/test_metrics.py::test_sharpness_blurry_image -v`
Expected: FAIL

- [ ] **Step 3: Implement sharpness metric**

Write `F:\AI\AItuning\scorer\metrics\sharpness.py`:

```python
import numpy as np
from scipy.optimize import curve_fit
from ..preprocess.pipeline import PreprocessedImage
from ..metrics import MetricResult, register


@register
def sharpness(pp: PreprocessedImage) -> MetricResult:
    """Score sharpness by analyzing edge profile widths and gradient statistics.

    Uses gradient magnitude for a global sharpness estimate and edge profile
    fitting for MTF-like measurement. Higher gradient mean = sharper.
    """
    y = pp.y
    grad = pp.gradient_mag

    # Global sharpness: mean gradient normalized
    mean_grad = float(grad.mean())
    # Typical mean gradient for a sharp image ~ 15-25 (in uint8 units)
    sharpness_raw = mean_grad / 20.0  # normalized
    score = min(100.0, sharpness_raw * 100.0)

    # Estimate edge width from gradient profile (simplified MTF-like)
    edge_width = _estimate_edge_width(grad)

    heatmap = 1.0 / (grad + 1.0)  # Low values = sharper (inverted for display)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = (heatmap * (100 - score) / 100.0).astype(np.float32)

    region_scores = {}
    for name, mask in pp.roi.items():
        g = grad[mask]
        if g.size > 0:
            raw = float(g.mean()) / 20.0
            region_scores[name] = float(min(100.0, raw * 100.0))
        else:
            region_scores[name] = 0.0

    parts = []
    # Find worst ROI
    center_score = region_scores.get("center", 0)
    edge_scores = [v for k, v in region_scores.items() if "edge" in k]
    corner_scores = [v for k, v in region_scores.items() if "corner" in k]

    if center_score < 60:
        parts.append(f"中心区域清晰度不足 (得分{center_score:.1f})")
    if edge_scores and np.mean(edge_scores) < 60:
        parts.append(f"边缘区域清晰度不足 (平均{np.mean(edge_scores):.1f})")
    if corner_scores and np.mean(corner_scores) < 50:
        parts.append(f"角落区域清晰度明显下降 (平均{np.mean(corner_scores):.1f})")

    diagnosis = "；".join(parts) if parts else "清晰度正常"

    return MetricResult(
        name="sharpness",
        global_score=round(score, 1),
        heatmap=heatmap,
        region_scores=region_scores,
        diagnosis=diagnosis,
        metadata={"mean_gradient": round(mean_grad, 2),
                  "edge_width_px": round(edge_width, 3)},
    )


def _estimate_edge_width(grad: np.ndarray) -> float:
    """Estimate average edge width from gradient magnitude.

    Takes the top 5% gradient pixels as edge candidates, measures profile width.
    """
    threshold = np.percentile(grad, 95)
    edge_pixels = grad >= threshold
    if edge_pixels.sum() < 10:
        return 10.0  # very soft

    # Distance transform: distance from each pixel to nearest non-edge
    from scipy.ndimage import distance_transform_edt
    dist_to_non_edge = distance_transform_edt(edge_pixels)
    # Distance from each non-edge pixel to nearest edge
    dist_to_edge = distance_transform_edt(~edge_pixels)

    # Edge width ≈ avg distance to transition (in pixels)
    near_edge = (dist_to_edge > 0) & (dist_to_edge < 5)
    if near_edge.sum() > 0:
        return float(2.0 * dist_to_edge[near_edge].mean())
    return 10.0
```

- [ ] **Step 4: Implement noise metric**

Write `F:\AI\AItuning\scorer\metrics\noise.py`:

```python
import numpy as np
from ..preprocess.pipeline import PreprocessedImage
from ..metrics import MetricResult, register


@register
def noise(pp: PreprocessedImage) -> MetricResult:
    """Score noise using SNR estimation and ISO 15739 visual noise approach.

    Luma noise measured on Y channel in flat regions. Chroma noise on Cb/Cr.
    Higher SNR = higher score.
    """
    y = pp.y
    cb = pp.cb
    cr = pp.cr

    # Use flat regions for noise estimation
    if pp.flat_mask.sum() > 100:
        y_flat = y[pp.flat_mask]
        cb_flat = cb[pp.flat_mask]
        cr_flat = cr[pp.flat_mask]
    else:
        y_flat = y.ravel()
        cb_flat = cb.ravel()
        cr_flat = cr.ravel()

    # Luma noise (standard deviation in flat regions)
    luma_noise = float(np.std(y_flat))

    # Chroma noise
    chroma_noise = float(np.std(cb_flat) + np.std(cr_flat))

    # SNR = signal / noise (signal ≈ mean of flat regions)
    signal = float(np.mean(y_flat))
    if luma_noise > 0:
        snr_db = float(20 * np.log10(signal / luma_noise))
    else:
        snr_db = 60.0  # essentially noiseless

    # Score: SNR > 40dB is excellent, < 20dB is poor
    snr_score = max(0.0, min(100.0, (snr_db - 15.0) / 30.0 * 100.0))
    chroma_penalty = min(40.0, chroma_noise * 200.0)
    score = max(0.0, snr_score - chroma_penalty)

    # Heatmap: local std in sliding windows
    from scipy.ndimage import uniform_filter
    h, w = y.shape
    local_mean = uniform_filter(y, size=8)
    local_sq_mean = uniform_filter(y ** 2, size=8)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))
    heatmap = local_std.astype(np.float32)

    region_scores = {}
    for name, mask in pp.roi.items():
        ym = y[mask]
        if ym.size > 0 and np.std(ym) > 0:
            sig = float(np.mean(ym))
            r_snr = float(20 * np.log10(sig / np.std(ym)))
            region_scores[name] = float(max(0.0, min(100.0, (r_snr - 15.0) / 30.0 * 100.0)))
        else:
            region_scores[name] = 100.0

    parts = []
    if luma_noise > 0.02:
        parts.append(f"亮度噪声偏高 (std={luma_noise:.4f})")
    if chroma_noise > 0.015:
        parts.append(f"色度噪声偏高 (chroma_std={chroma_noise:.4f})")
    if snr_db < 25:
        parts.append(f"信噪比偏低 ({snr_db:.1f}dB)")

    diagnosis = "；".join(parts) if parts else "噪声控制良好"

    return MetricResult(
        name="noise",
        global_score=round(score, 1),
        heatmap=heatmap,
        region_scores=region_scores,
        diagnosis=diagnosis,
        metadata={"snr_db": round(snr_db, 2), "luma_noise_std": round(luma_noise, 4),
                  "chroma_noise_std": round(chroma_noise, 4)},
    )
```

- [ ] **Step 5: Run sharpness and noise tests**

Run: `F:/AI/imglab/.venv/Scripts/python.exe -m pytest F:/AI/AItuning/tests/test_metrics.py::test_sharpness_blurry_image F:/AI/AItuning/tests/test_metrics.py::test_noise_uniform_image F:/AI/AItuning/tests/test_metrics.py::test_noise_noisy_image -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add F:/AI/AItuning/scorer/metrics/sharpness.py F:/AI/AItuning/scorer/metrics/noise.py F:/AI/AItuning/tests/test_metrics.py
git commit -m "feat: sharpness and noise metrics"
```

---

### Task 10: Remaining Metrics — Color, White Balance, Dynamic Range, Texture, Uniformity, Fringing

**Files:**
- Create: `F:\AI\AItuning\scorer\metrics\color_accuracy.py`
- Create: `F:\AI\AItuning\scorer\metrics\white_balance.py`
- Create: `F:\AI\AItuning\scorer\metrics\dynamic_range.py`
- Create: `F:\AI\AItuning\scorer\metrics\texture.py`
- Create: `F:\AI\AItuning\scorer\metrics\uniformity.py`
- Create: `F:\AI\AItuning\scorer\metrics\fringing.py`
- Modify: `F:\AI\AItuning\tests\test_metrics.py`

- [ ] **Step 1: Write color/white balance/DR/texture/uniformity/fringing tests**

Append to `F:\AI\AItuning\tests\test_metrics.py`:

```python
from scorer.metrics.color_accuracy import color_accuracy
from scorer.metrics.white_balance import white_balance
from scorer.metrics.dynamic_range import dynamic_range
from scorer.metrics.texture import texture_preservation
from scorer.metrics.uniformity import uniformity
from scorer.metrics.fringing import fringing


def test_color_accuracy_neutral():
    """Neutral image (Cb=Cr=0) gets high color accuracy score."""
    pp = make_preprocessed(cb_array=np.zeros((64, 64), dtype=np.float32),
                           cr_array=np.zeros((64, 64), dtype=np.float32))
    result = color_accuracy(pp)
    assert result.global_score > 60
    assert "delta_c" in result.metadata


def test_white_balance_gray_world():
    """Image satisfying gray-world assumption gets high WB score."""
    pp = make_preprocessed(cb_array=np.zeros((64, 64), dtype=np.float32),
                           cr_array=np.zeros((64, 64), dtype=np.float32))
    result = white_balance(pp)
    assert result.global_score > 60


def test_dynamic_range():
    """Full-range image gets reasonable DR score."""
    y = np.linspace(0.02, 0.98, 64 * 64).reshape(64, 64).astype(np.float32)
    pp = make_preprocessed(y_array=y)
    result = dynamic_range(pp)
    assert result.global_score > 30
    assert "dr_stops" in result.metadata


def test_texture_preservation():
    """Image with detail zone gets texture score."""
    y = np.full((64, 64), 0.5, dtype=np.float32)
    y[16:48, 16:48] += np.random.randn(32, 32).astype(np.float32) * 0.02
    pp = make_preprocessed(y_array=y)
    result = texture_preservation(pp)
    assert 0 <= result.global_score <= 100


def test_uniformity_perfect():
    """Uniform image gets high uniformity score."""
    pp = make_preprocessed()
    result = uniformity(pp)
    assert result.global_score > 70
    assert "corner_vs_center_luma_ratio" in result.metadata


def test_fringing_none():
    """Image with no edges has no color fringing."""
    pp = make_preprocessed()
    result = fringing(pp)
    assert result.global_score > 90  # no edges = no fringing
```

- [ ] **Step 2: Run one test to confirm failure**

Run: `F:/AI/imglab/.venv/Scripts/python.exe -m pytest F:/AI/AItuning/tests/test_metrics.py::test_color_accuracy_neutral -v`
Expected: FAIL

- [ ] **Step 3: Implement color_accuracy**

Write `F:\AI\AItuning\scorer\metrics\color_accuracy.py`:

```python
import numpy as np
from ..preprocess.pipeline import PreprocessedImage
from ..metrics import MetricResult, register


@register
def color_accuracy(pp: PreprocessedImage) -> MetricResult:
    """Score color accuracy via chroma channel deviation from gray-world neutral.

    Assumes scenes average to gray (gray-world assumption). Large mean |Cb| or |Cr|
    indicates a color cast or over-saturation.
    """
    cb, cr = pp.cb, pp.cr
    delta_cb = float(np.mean(np.abs(cb)))
    delta_cr = float(np.mean(np.abs(cr)))
    delta_c = np.sqrt(delta_cb ** 2 + delta_cr ** 2)

    # Score: delta_c < 0.02 is good, > 0.10 is poor
    score = max(0.0, 100.0 - delta_c * 800.0)

    # Heatmap: chroma magnitude per pixel
    heatmap = (np.abs(cb) + np.abs(cr)).astype(np.float32)

    region_scores = {}
    for name, mask in pp.roi.items():
        dc = float(np.sqrt(np.mean(cb[mask]) ** 2 + np.mean(cr[mask]) ** 2))
        region_scores[name] = float(max(0.0, 100.0 - dc * 800.0))

    parts = []
    if delta_cb > 0.03:
        direction = "偏蓝" if delta_cb > 0 else "偏黄"
        parts.append(f"Cb通道偏移 ({direction}, ΔCb={delta_cb:.3f})")
    if delta_cr > 0.03:
        direction = "偏红" if delta_cr > 0 else "偏青"
        parts.append(f"Cr通道偏移 ({direction}, ΔCr={delta_cr:.3f})")

    diagnosis = "；".join(parts) if parts else "色彩还原正常"

    return MetricResult(
        name="color_accuracy",
        global_score=round(score, 1),
        heatmap=heatmap,
        region_scores=region_scores,
        diagnosis=diagnosis,
        metadata={"delta_cb": round(delta_cb, 4), "delta_cr": round(delta_cr, 4),
                  "delta_c": round(delta_c, 4)},
    )
```

- [ ] **Step 4: Implement white_balance**

Write `F:\AI\AItuning\scorer\metrics\white_balance.py`:

```python
import numpy as np
from ..preprocess.pipeline import PreprocessedImage
from ..metrics import MetricResult, register


@register
def white_balance(pp: PreprocessedImage) -> MetricResult:
    """Score white balance using gray-world assumption and Cb/Cr neutrality.

    The mean Cb and Cr should be near 0 for a well-white-balanced image.
    Also checks white-patch (top 5% brightest pixels should be neutral).
    """
    y, cb, cr = pp.y, pp.cb, pp.cr

    # Gray-world: mean chroma should be 0
    cb_mean = float(cb.mean())
    cr_mean = float(cr.mean())
    gw_offset = np.sqrt(cb_mean ** 2 + cr_mean ** 2)

    # White-patch: top 5% brightest pixels should have near-zero chroma
    threshold = np.percentile(y, 95)
    bright_mask = y >= threshold
    if bright_mask.sum() > 10:
        wp_cb = float(cb[bright_mask].mean())
        wp_cr = float(cr[bright_mask].mean())
        wp_offset = np.sqrt(wp_cb ** 2 + wp_cr ** 2)
    else:
        wp_offset = 0.0

    # Score based on combined offset
    total_offset = 0.7 * gw_offset + 0.3 * wp_offset
    score = max(0.0, 100.0 - total_offset * 500.0)

    # Estimated color temperature (simplified: Cb/Cr ratio)
    if abs(cr_mean) > 1e-6:
        estimated_k = 5000.0 + cb_mean / (cr_mean + 1e-6) * 3000.0
    else:
        estimated_k = 6500.0

    heatmap = np.sqrt(cb ** 2 + cr ** 2).astype(np.float32)

    region_scores = {}
    for name, mask in pp.roi.items():
        cm = float(np.sqrt(cb[mask].mean() ** 2 + cr[mask].mean() ** 2))
        region_scores[name] = float(max(0.0, 100.0 - cm * 500.0))

    parts = []
    if gw_offset > 0.02:
        parts.append(f"灰世界偏离 (offset={gw_offset:.3f})")
    if wp_offset > 0.02:
        parts.append(f"白点不中性 (offset={wp_offset:.3f})")

    diagnosis = "；".join(parts) if parts else "白平衡正常"

    return MetricResult(
        name="white_balance",
        global_score=round(score, 1),
        heatmap=heatmap,
        region_scores=region_scores,
        diagnosis=diagnosis,
        metadata={"estimated_illuminant_k": round(estimated_k, 0),
                  "gray_world_offset": round(gw_offset, 4),
                  "white_patch_offset": round(wp_offset, 4)},
    )
```

- [ ] **Step 5: Implement dynamic_range**

Write `F:\AI\AItuning\scorer\metrics\dynamic_range.py`:

```python
import numpy as np
from ..preprocess.pipeline import PreprocessedImage
from ..metrics import MetricResult, register


@register
def dynamic_range(pp: PreprocessedImage) -> MetricResult:
    """Score dynamic range from histogram spread and clipping behavior.

    Estimates DR in stops from the ratio of 99th to 1st percentile (excluding
    clipped blacks/whites). Also detects highlight headroom.
    """
    y = pp.y
    y_valid = y[(y > 0.005) & (y < 0.995)]

    if y_valid.size > 100:
        p01 = float(np.percentile(y_valid, 1))
        p99 = float(np.percentile(y_valid, 99))
        if p01 > 0:
            dr_stops = float(np.log2(p99 / p01))
        else:
            dr_stops = float(np.log2(p99 / 0.001))
    else:
        dr_stops = 0.0

    highlight_clip = float((y > 0.98).mean())
    shadow_clip = float((y < 0.02).mean())
    headroom_stops = float(-np.log2(max(highlight_clip, 0.001)))

    # Score: 8+ stops is excellent, < 4 is poor
    score = max(0.0, min(100.0, (dr_stops - 3.0) / 6.0 * 100.0))
    clip_penalty = (highlight_clip + shadow_clip) * 200.0
    score = max(0.0, score - clip_penalty)

    heatmap = (1.0 - pp.gradient_mag / (pp.gradient_mag.max() + 1e-8)).astype(np.float32)

    region_scores = {}
    for name, mask in pp.roi.items():
        ym = y[mask]
        yv = ym[(ym > 0.005) & (ym < 0.995)]
        if yv.size > 10 and np.percentile(yv, 1) > 0:
            dr = float(np.log2(np.percentile(yv, 99) / np.percentile(yv, 1)))
            region_scores[name] = float(max(0.0, min(100.0, (dr - 3.0) / 6.0 * 100.0)))
        else:
            region_scores[name] = 0.0

    parts = []
    if dr_stops < 5:
        parts.append(f"动态范围偏低 ({dr_stops:.1f} stops)")
    if highlight_clip > 0.03:
        parts.append(f"高光裁切严重 ({highlight_clip:.1%})")
    if shadow_clip > 0.03:
        parts.append(f"暗部裁切严重 ({shadow_clip:.1%})")

    diagnosis = "；".join(parts) if parts else "动态范围正常"

    return MetricResult(
        name="dynamic_range",
        global_score=round(score, 1),
        heatmap=heatmap,
        region_scores=region_scores,
        diagnosis=diagnosis,
        metadata={"dr_stops": round(dr_stops, 2),
                  "headroom_stops": round(headroom_stops, 2)},
    )
```

- [ ] **Step 6: Implement texture_preservation**

Write `F:\AI\AItuning\scorer\metrics\texture.py`:

```python
import numpy as np
from ..preprocess.pipeline import PreprocessedImage
from ..metrics import MetricResult, register


@register
def texture_preservation(pp: PreprocessedImage) -> MetricResult:
    """Score texture preservation: ratio of variance in detail zones vs flat zones.

    Higher ratio = better texture differentiation = more detail preserved.
    """
    y = pp.y

    detail_var = float(np.var(y[pp.detail_mask])) if pp.detail_mask.sum() > 10 else 0.0
    flat_var = float(np.var(y[pp.flat_mask])) if pp.flat_mask.sum() > 10 else 1e-6

    if flat_var > 0:
        ratio = detail_var / flat_var
    else:
        ratio = detail_var / 1e-6 if detail_var > 0 else 1.0

    # Score: ratio > 5 is good (strong texture separation), < 1.5 is poor
    score = max(0.0, min(100.0, (ratio - 1.0) / 8.0 * 100.0))

    heatmap = np.zeros_like(y, dtype=np.float32)
    heatmap[pp.detail_mask] = 0.0
    heatmap[pp.flat_mask] = 1.0

    region_scores = {}
    for name, mask in pp.roi.items():
        detail_in_roi = mask & pp.detail_mask
        flat_in_roi = mask & pp.flat_mask
        dv = float(np.var(y[detail_in_roi])) if detail_in_roi.sum() > 10 else 0.0
        fv = float(np.var(y[flat_in_roi])) if flat_in_roi.sum() > 10 else 1e-6
        r = dv / fv if fv > 0 else 1.0
        region_scores[name] = float(max(0.0, min(100.0, (r - 1.0) / 8.0 * 100.0)))

    parts = []
    if ratio < 2.0:
        parts.append(f"纹理/平坦方差比偏低 ({ratio:.2f})，细节分辨力不足")

    diagnosis = "；".join(parts) if parts else "纹理保留正常"

    return MetricResult(
        name="texture_preservation",
        global_score=round(score, 1),
        heatmap=heatmap,
        region_scores=region_scores,
        diagnosis=diagnosis,
        metadata={"texture_to_flat_variance_ratio": round(ratio, 2)},
    )
```

- [ ] **Step 7: Implement uniformity**

Write `F:\AI\AItuning\scorer\metrics\uniformity.py`:

```python
import numpy as np
from ..preprocess.pipeline import PreprocessedImage
from ..metrics import MetricResult, register


@register
def uniformity(pp: PreprocessedImage) -> MetricResult:
    """Score uniformity (lens shading) via corner-to-center luminance ratio.

    For each corner, compares mean Y to center mean Y. Ideal ratio ≈ 0.9+.
    Also checks color shading via Cb/Cr corner-to-center differences.
    """
    y = pp.y
    center_mask = pp.roi["center"]
    center_y = float(y[center_mask].mean()) if center_mask.sum() > 0 else 0.5

    corner_names = ["corner_tl", "corner_tr", "corner_bl", "corner_br"]
    ratios = []
    for cn in corner_names:
        mask = pp.roi[cn]
        if mask.sum() > 0 and center_y > 0:
            ratios.append(float(y[mask].mean()) / center_y)

    if ratios:
        mean_ratio = float(np.mean(ratios))
        worst_ratio = float(np.min(ratios))
    else:
        mean_ratio = 1.0
        worst_ratio = 1.0

    # Score: ratio > 0.85 is good, < 0.60 is poor
    score = max(0.0, min(100.0, (worst_ratio - 0.5) / 0.45 * 100.0))

    # Heatmap: luminance normalized by center
    heatmap = np.abs(y - center_y).astype(np.float32)

    region_scores = {}
    for name, mask in pp.roi.items():
        if mask.sum() > 0 and center_y > 0:
            r = float(y[mask].mean()) / center_y
            region_scores[name] = float(max(0.0, min(100.0, (r - 0.5) / 0.45 * 100.0)))
        else:
            region_scores[name] = 100.0

    parts = []
    if worst_ratio < 0.80:
        parts.append(f"角落亮度衰减明显 (最差角落/中心比={worst_ratio:.2f})")
        # Find worst corner
        worst_cn = corner_names[np.argmin(ratios)] if ratios else "unknown"
        parts.append(f"最严重区域: {worst_cn}")
    if mean_ratio < 0.85:
        parts.append(f"整体均匀性不足 (平均比={mean_ratio:.2f})")

    diagnosis = "；".join(parts) if parts else "画面均匀性良好"

    return MetricResult(
        name="uniformity",
        global_score=round(score, 1),
        heatmap=heatmap,
        region_scores=region_scores,
        diagnosis=diagnosis,
        metadata={"corner_vs_center_luma_ratio": round(worst_ratio, 3),
                  "mean_corner_ratio": round(mean_ratio, 3)},
    )
```

- [ ] **Step 8: Implement fringing**

Write `F:\AI\AItuning\scorer\metrics\fringing.py`:

```python
import numpy as np
from ..preprocess.pipeline import PreprocessedImage
from ..metrics import MetricResult, register


@register
def fringing(pp: PreprocessedImage) -> MetricResult:
    """Detect color fringing (purple/cyan) near high-contrast edges.

    Purple fringing appears as strong positive Cr AND negative Cb near edges.
    """
    cb, cr = pp.cb, pp.cr
    edge = pp.edge_mask

    if edge.sum() < 10:
        return MetricResult(
            name="fringing",
            global_score=100.0,
            heatmap=np.zeros_like(cb, dtype=np.float32),
            region_scores={name: 100.0 for name in pp.roi},
            diagnosis="无明显紫边（边缘像素不足）",
            metadata={"fringe_pixel_ratio": 0.0},
        )

    # Dilate edge to include adjacent pixels
    from scipy.ndimage import binary_dilation
    edge_zone = binary_dilation(edge, iterations=2)

    # Purple fringing: Cr positive AND Cb negative (magenta in YCbCr)
    purple = (cr > 0.05) & (cb < -0.03) & edge_zone

    # Cyan fringing: Cr negative AND Cb positive (cyan/green in YCbCr)
    cyan = (cr < -0.03) & (cb > 0.03) & edge_zone

    fringe = purple | cyan
    fringe_ratio = float(fringe.sum()) / max(edge_zone.sum(), 1)

    # Score: < 0.5% fringe is excellent, > 5% is poor
    score = max(0.0, 100.0 - fringe_ratio * 100.0 * 15.0)

    heatmap = (fringe.astype(np.float32) * (100.0 - score) / 100.0)

    region_scores = {}
    for name, mask in pp.roi.items():
        zone = edge_zone & mask
        fz = fringe & mask
        ratio = float(fz.sum()) / max(zone.sum(), 1)
        region_scores[name] = float(max(0.0, 100.0 - ratio * 100.0 * 15.0))

    parts = []
    if fringe_ratio > 0.01:
        n_purple = int(purple.sum())
        n_cyan = int(cyan.sum())
        parts.append(f"检测到色散边纹 (紫边{n_purple}px, 青边{n_cyan}px")
        parts.append(f"色散占比 {fringe_ratio:.2%}")
    # Find worst region
    worst_region = min(region_scores, key=region_scores.get) if region_scores else "unknown"
    if region_scores.get(worst_region, 100) < 80:
        parts.append(f"最严重区域: {worst_region}")

    diagnosis = "；".join(parts) if parts else "无明显紫边/色散"

    return MetricResult(
        name="fringing",
        global_score=round(score, 1),
        heatmap=heatmap,
        region_scores=region_scores,
        diagnosis=diagnosis,
        metadata={"fringe_pixel_ratio": round(fringe_ratio, 4)},
    )
```

- [ ] **Step 9: Run all metrics tests, verify get_all_metrics returns 11**

Run: `F:/AI/imglab/.venv/Scripts/python.exe -m pytest F:/AI/AItuning/tests/test_metrics.py -v`
Expected: all PASS, including `test_get_all_metrics_returns_11`

- [ ] **Step 10: Commit**

```bash
git add F:/AI/AItuning/scorer/metrics/color_accuracy.py F:/AI/AItuning/scorer/metrics/white_balance.py F:/AI/AItuning/scorer/metrics/dynamic_range.py F:/AI/AItuning/scorer/metrics/texture.py F:/AI/AItuning/scorer/metrics/uniformity.py F:/AI/AItuning/scorer/metrics/fringing.py F:/AI/AItuning/tests/test_metrics.py
git commit -m "feat: color accuracy, white balance, dynamic range, texture, uniformity, fringing metrics"
```

---

### Task 11: Aggregation

**Files:**
- Create: `F:\AI\AItuning\scorer\aggregate.py`

- [ ] **Step 1: Implement aggregation**

Write `F:\AI\AItuning\scorer\aggregate.py`:

```python
from .metrics import MetricResult


DEFAULT_WEIGHTS = {
    "exposure": 1.0,
    "brightness": 0.8,
    "contrast": 1.0,
    "color_accuracy": 1.2,
    "white_balance": 1.0,
    "sharpness": 1.5,
    "noise": 1.2,
    "dynamic_range": 1.0,
    "texture_preservation": 1.0,
    "uniformity": 0.8,
    "fringing": 0.5,
}


def compute_total_score(results: list[MetricResult],
                        weights: dict[str, float] | None = None) -> float:
    """Weighted total score from individual metric results."""
    if weights is None:
        weights = DEFAULT_WEIGHTS
    total_weight = 0.0
    weighted_sum = 0.0
    for r in results:
        w = weights.get(r.name, 1.0)
        weighted_sum += r.global_score * w
        total_weight += w
    return round(weighted_sum / total_weight, 1) if total_weight > 0 else 0.0


def compute_deltas(results_test: list[MetricResult],
                   results_ref: list[MetricResult]) -> list[dict]:
    """Per-metric delta between test and reference device."""
    ref_map = {r.name: r for r in results_ref}
    deltas = []
    for r in results_test:
        ref = ref_map.get(r.name)
        if ref:
            deltas.append({
                "metric": r.name,
                "test_score": r.global_score,
                "ref_score": ref.global_score,
                "delta": round(r.global_score - ref.global_score, 1),
            })
    return deltas


def flag_issues(results: list[MetricResult], threshold: float = 60.0) -> list[MetricResult]:
    """Return metrics that scored below threshold."""
    return [r for r in results if r.global_score < threshold]
```

- [ ] **Step 2: Commit**

```bash
git add F:/AI/AItuning/scorer/aggregate.py
git commit -m "feat: weighted aggregation, comparison deltas, issue flagging"
```

---

### Task 12: Console Report & Visual Report

**Files:**
- Create: `F:\AI\AItuning\scorer\report\console.py`
- Create: `F:\AI\AItuning\scorer\report\render.py`

- [ ] **Step 1: Implement console report**

Write `F:\AI\AItuning\scorer\report\console.py`:

```python
from ..metrics import MetricResult
from ..aggregate import compute_total_score, flag_issues


def print_report(results: list[MetricResult], total_score: float | None = None):
    """Print formatted quality report to console."""
    if total_score is None:
        total_score = compute_total_score(results)

    print("=" * 70)
    print(f"  图像质量打分报告 (总分: {total_score:.1f}/100)")
    print("=" * 70)
    print(f"{'指标':<16} {'分数':>6} {'状态':<10}")
    print("-" * 70)

    issues = flag_issues(results)
    for r in results:
        status = "⚠️ 待优化" if r.global_score < 60 else "✓ 正常"
        print(f"  {r.name:<14} {r.global_score:>6.1f}  {status:<10}")

    print("=" * 70)

    if issues:
        print("\n🔍 问题诊断:")
        for r in issues:
            print(f"  [{r.name}] 得分 {r.global_score:.1f}")
            print(f"    → {r.diagnosis}")
        print()

    print("💡 调试建议:")
    _print_recommendations(results)


def _print_recommendations(results: list[MetricResult]):
    score_map = {r.name: r.global_score for r in results}

    if score_map.get("sharpness", 100) < 60:
        # Find region with worst sharpness
        for r in results:
            if r.name == "sharpness" and r.region_scores:
                worst = min(r.region_scores, key=r.region_scores.get)
                print(f"  清晰度: 检查 {worst} 区域, 可能需要调整镜头对焦或锐化参数")

    if score_map.get("noise", 100) < 60:
        print(f"  噪声: 考虑降低ISO, 或调整降噪强度(注意平衡纹理保留)")

    if score_map.get("white_balance", 100) < 60:
        print(f"  白平衡: 检查AWB算法色温估计, 可能存在色温偏差")

    if score_map.get("exposure", 100) < 60:
        print(f"  曝光: 检查AE target设置, 目标Y均值应接近0.45-0.55")

    if score_map.get("uniformity", 100) < 60:
        print(f"  均匀性: 检查镜头shading校正, 角落亮度不应低于中心80%")
```

- [ ] **Step 2: Implement visual report renderer**

Write `F:\AI\AItuning\scorer\report\render.py`:

```python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ..metrics import MetricResult
from ..aggregate import compute_total_score


def render_report(results: list[MetricResult], total_score: float,
                  image_y: np.ndarray | None, output_path: str):
    """Render visual quality report to PNG.

    Layout: radar chart (left) + 3×4 heatmap grid (right) + diagnosis text at bottom.
    """
    fig = plt.figure(figsize=(20, 18))

    # ── Left: Radar chart ──
    ax_radar = fig.add_axes([0.02, 0.55, 0.28, 0.40], polar=True)
    _draw_radar(ax_radar, results, total_score)

    # ── Right: Heatmap grid (3 rows × 4 cols) ──
    _draw_heatmap_grid(fig, results, image_y)

    # ── Bottom: Diagnosis text ──
    ax_text = fig.add_axes([0.05, 0.02, 0.90, 0.12])
    ax_text.axis("off")
    _draw_diagnosis_text(ax_text, results, total_score)

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _draw_radar(ax, results, total_score):
    """Polar radar chart of all 11 metrics."""
    names = [r.name for r in results]
    scores = [r.global_score for r in results]
    N = len(names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    scores_plot = scores + [scores[0]]
    angles_plot = angles + [angles[0]]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=6)
    ax.fill(angles_plot, scores_plot, alpha=0.25, color="#2196F3")
    ax.plot(angles_plot, scores_plot, color="#2196F3", linewidth=2)
    ax.set_title(f"总分: {total_score:.1f}/100", fontsize=12, fontweight="bold", pad=20)


def _draw_heatmap_grid(fig, results, image_y):
    """Draw 3×4 grid of heatmaps, each overlaid on a downsized Y preview."""
    for idx, r in enumerate(results):
        ax = fig.add_axes([0.35 + (idx % 4) * 0.16, 0.55 - (idx // 4) * 0.30, 0.13, 0.12])
        hm = r.heatmap
        if hm is not None and hm.size > 1:
            ax.imshow(hm, cmap="hot", interpolation="nearest", aspect="auto")
            ax.set_title(f"{r.name}\n{r.global_score:.1f}", fontsize=8)
        else:
            ax.text(0.5, 0.5, f"{r.global_score:.1f}", ha="center", va="center", fontsize=14,
                    transform=ax.transAxes)
            ax.set_title(r.name, fontsize=8)
        ax.axis("off")

    # Hide unused subplot positions (only 11 metrics need 1 hidden)
    # Grid is 3×4=12, hide cell 12
    ax = fig.add_axes([0.35 + 3 * 0.16, 0.55 - 2 * 0.30, 0.13, 0.12])
    ax.axis("off")


def _draw_diagnosis_text(ax, results, total_score):
    """Draw text summary of issues flagged."""
    lines = []
    lines.append(f"综合总分: {total_score:.1f}/100")
    for r in results:
        if r.global_score < 60:
            lines.append(f"⚠ {r.name}: {r.global_score:.1f} — {r.diagnosis}")

    if len(lines) == 1:
        lines.append("所有指标均在正常范围，图像质量合格。")

    text = "\n".join(lines)
    ax.text(0.01, 0.5, text, fontsize=10, fontfamily="monospace",
            verticalalignment="center", transform=ax.transAxes)
```

- [ ] **Step 3: Commit**

```bash
git add F:/AI/AItuning/scorer/report/console.py F:/AI/AItuning/scorer/report/render.py
git commit -m "feat: console report and visual report renderer"
```

---

### Task 13: Pipeline Orchestrator & CLI

**Files:**
- Create: `F:\AI\AItuning\scorer\pipeline.py`
- Create: `F:\AI\AItuning\run.py`

- [ ] **Step 1: Implement pipeline orchestrator**

Write `F:\AI\AItuning\scorer\pipeline.py`:

```python
"""Orchestrates the full 5-layer pipeline."""
from .io.nv12 import read_nv12
from .io.png_reader import read_png
from .io.image_model import Image
from .preprocess.pipeline import preprocess, PreprocessedImage
from .metrics import get_all_metrics, MetricResult
from .aggregate import compute_total_score, compute_deltas


def run_pipeline(image_path: str, width: int | None = None, height: int | None = None,
                 ref_path: str | None = None) -> dict:
    """Run full quality scoring pipeline.

    Returns:
        dict with: results, total_score, ref_results, ref_total_score, deltas, image
    """
    # Layer 1: I/O
    img = _read_image(image_path, width, height)

    # Layer 2: Preprocessing
    pp = preprocess(img)

    # Layer 3: Metrics
    metrics = get_all_metrics()
    results = [metric(pp) for metric in metrics]

    # Layer 4: Aggregation
    total_score = compute_total_score(results)

    output = {
        "results": results,
        "total_score": total_score,
        "ref_results": None,
        "ref_total_score": None,
        "deltas": None,
        "image_y": img.y,
    }

    # Reference device comparison
    if ref_path:
        ref_img = _read_image(ref_path, width, height)
        ref_pp = preprocess(ref_img)
        ref_results = [metric(ref_pp) for metric in metrics]
        ref_total = compute_total_score(ref_results)
        deltas = compute_deltas(results, ref_results)
        output["ref_results"] = ref_results
        output["ref_total_score"] = ref_total
        output["deltas"] = deltas

    return output


def _read_image(path: str, width: int | None, height: int | None) -> Image:
    """Read image, auto-detecting format from extension."""
    import os
    ext = os.path.splitext(path)[1].lower()
    if ext == ".nv12":
        if width is None or height is None:
            raise ValueError("NV12 requires --width and --height")
        return read_nv12(path, width, height)
    elif ext == ".png":
        return read_png(path)
    else:
        raise ValueError(f"Unsupported format: {ext}. Use .nv12 or .png")
```

- [ ] **Step 2: Implement CLI entry point**

Write `F:\AI\AItuning\run.py`:

```python
#!/usr/bin/env python3
"""Image Quality Scoring System — CLI entry point.

Usage:
    python run.py <image_path> [--width W] [--height H] [--ref <ref_image>]
"""
import argparse
import sys
from scorer.pipeline import run_pipeline
from scorer.report.console import print_report
from scorer.report.render import render_report


def main():
    parser = argparse.ArgumentParser(description="Image Quality Scoring System")
    parser.add_argument("image", help="Path to input image (.nv12 or .png)")
    parser.add_argument("--width", type=int, help="Image width (required for .nv12)")
    parser.add_argument("--height", type=int, help="Image height (required for .nv12)")
    parser.add_argument("--ref", help="Path to reference device image (optional)")
    parser.add_argument("--output", "-o", default="report.png",
                        help="Output report image path (default: report.png)")
    args = parser.parse_args()

    try:
        output = run_pipeline(args.image, args.width, args.height, args.ref)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Print console report
    print_report(output["results"], output["total_score"])

    if output["ref_results"]:
        print("\n" + "=" * 70)
        print("  对比机 vs 待测机 分数差异")
        print("=" * 70)
        for d in output["deltas"]:
            direction = "↑ 优于" if d["delta"] > 0 else ("↓ 差于" if d["delta"] < 0 else "= 持平")
            print(f"  {d['metric']:<14} 待测:{d['test_score']:>5.1f}  对比:{d['ref_score']:>5.1f}  "
                  f"Δ:{d['delta']:>+6.1f} {direction}")
        print(f"\n  待测机总分: {output['total_score']:.1f}  |  对比机总分: {output['ref_total_score']:.1f}")

    # Render visual report
    render_report(output["results"], output["total_score"],
                  output["image_y"], args.output)
    print(f"\n📊 可视化报告已保存至: {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify CLI runs (help)**

Run: `F:/AI/imglab/.venv/Scripts/python.exe F:/AI/AItuning/run.py --help`
Expected: Prints help text with all arguments

- [ ] **Step 4: Commit**

```bash
git add F:/AI/AItuning/scorer/pipeline.py F:/AI/AItuning/run.py
git commit -m "feat: pipeline orchestrator and CLI entry point"
```

---

### Task 14: Integration Test & Smoke Test

**Files:**
- Create: `F:\AI\AItuning\tests\test_integration.py`

- [ ] **Step 1: Write integration test with synthetic images**

Write `F:\AI\AItuning\tests\test_integration.py`:

```python
"""Integration tests: full pipeline on synthetic images."""
import numpy as np
import tempfile
import os
from scorer.pipeline import run_pipeline
from PIL import Image as PILImage


def _write_test_png(path, rgb_array):
    PILImage.fromarray((rgb_array * 255).astype(np.uint8)).save(path)


def test_pipeline_perfect_image():
    """Uniform mid-gray image should score high overall."""
    gray = np.full((64, 64, 3), 0.5, dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    try:
        _write_test_png(tmp, gray)
        output = run_pipeline(tmp)
        assert output["total_score"] > 55
        assert len(output["results"]) == 11
    finally:
        os.unlink(tmp)


def test_pipeline_degraded_image():
    """Noisy + blurry + color-cast image should score lower than perfect."""
    from scipy.ndimage import gaussian_filter

    gray = np.full((64, 64, 3), 0.5, dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        perfect_tmp = f.name
    try:
        _write_test_png(perfect_tmp, gray)
        perfect = run_pipeline(perfect_tmp)
    finally:
        os.unlink(perfect_tmp)

    # Create degraded image
    degraded = gray.copy()
    degraded += np.random.randn(64, 64, 3).astype(np.float32) * 0.08  # noise
    degraded[:, :, 0] = gaussian_filter(degraded[:, :, 0], sigma=1.5)  # blur
    degraded[:, :, 2] += 0.05  # blue cast
    degraded = np.clip(degraded, 0.0, 1.0)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        degraded_tmp = f.name
    try:
        _write_test_png(degraded_tmp, degraded)
        bad = run_pipeline(degraded_tmp)
    finally:
        os.unlink(degraded_tmp)

    assert perfect["total_score"] > bad["total_score"], \
        f"Perfect {perfect['total_score']} should beat degraded {bad['total_score']}"


def test_pipeline_with_ref():
    """Pipeline with reference image returns deltas."""
    gray = np.full((32, 32, 3), 0.5, dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f1, \
         tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f2:
        tmp_test = f1.name
        tmp_ref = f2.name
    try:
        _write_test_png(tmp_test, gray)
        _write_test_png(tmp_ref, gray)
        output = run_pipeline(tmp_test, ref_path=tmp_ref)
        assert output["deltas"] is not None
        assert output["ref_total_score"] is not None
        # Identical images should have near-zero deltas
        for d in output["deltas"]:
            assert abs(d["delta"]) < 5.0, f"{d['metric']} delta too large: {d['delta']}"
    finally:
        os.unlink(tmp_test)
        os.unlink(tmp_ref)


def test_nv12_pipeline():
    """Full pipeline with NV12 input."""
    w, h = 64, 64
    y_data = np.full((h, w), 128, dtype=np.uint8)
    uv_data = np.full((h // 2, w // 2, 2), 128, dtype=np.uint8)
    nv12_bytes = np.concatenate([y_data.flatten(), uv_data.flatten()])

    with tempfile.NamedTemporaryFile(suffix=".nv12", delete=False) as f:
        f.write(nv12_bytes.tobytes())
        tmp = f.name
    try:
        output = run_pipeline(tmp, width=w, height=h)
        assert output["total_score"] > 0
        assert len(output["results"]) == 11
    finally:
        os.unlink(tmp)
```

- [ ] **Step 2: Run integration tests**

Run: `F:/AI/imglab/.venv/Scripts/python.exe -m pytest F:/AI/AItuning/tests/test_integration.py -v`
Expected: all PASS (4 tests, may take a few seconds for matplotlib rendering)

- [ ] **Step 3: Run full test suite**

Run: `F:/AI/imglab/.venv/Scripts/python.exe -m pytest F:/AI/AItuning/tests/ -v`
Expected: all tests PASS

- [ ] **Step 4: Commit**

```bash
git add F:/AI/AItuning/tests/test_integration.py
git commit -m "test: integration tests covering perfect, degraded, NV12, and reference comparison"
```

---

## Final Verification

After all tasks complete:

```bash
# Run full test suite
F:/AI/imglab/.venv/Scripts/python.exe -m pytest F:/AI/AItuning/tests/ -v

# Smoke test with a real or synthetic PNG
F:/AI/imglab/.venv/Scripts/python.exe F:/AI/AItuning/run.py <some_test_image.png>
```
