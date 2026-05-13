# Image Quality Scoring System — Design Spec

**Date**: 2026-05-12
**Status**: Draft
**Scope**: Single-project, 11-metric image quality scoring pipeline

## 1. Purpose

A camera acceptance scoring system for image tuning engineers. Evaluates a single image (no-reference) across 11 quality dimensions, produces a 0–100 score per dimension with spatial diagnostics. Supports optional comparison against a reference device image.

## 2. Input / Output

| Direction | Format | Notes |
|-----------|--------|-------|
| Input | `.nv12` | YUV 4:2:0 semi-planar, width×height as metadata |
| Input | `.png` | RGB or RGBA, any bit depth |
| Input | comparison `.nv12`/`.png` | Optional reference device image |
| Output | Console report | Scores table, diagnosis text |
| Output | `report.png` | Radar chart + heatmaps + ROI annotations |

## 3. Architecture — 5 Layer Pipeline

```
Layer 1: Image I/O        →  parse raw pixels into float32 ndarray
Layer 2: Preprocessing    →  YCbCr conversion, ROI masks, edge/texture/flat maps
Layer 3: Metric Modules   →  11 independent metric functions, each returns score+heatmap+diagnosis
Layer 4: Aggregation      →  weighted total, comparison deltas
Layer 5: Report           →  radar chart, heatmap grid, diagnosis text dump
```

### 3.1 Layer 1 — Image I/O (`io/`)

- `nv12_reader(path, width, height)` → `(Y, Cb, Cr)` float32 ndarrays at full resolution (chroma upsampled via nearest-neighbor to match luma)
- `png_reader(path)` → `RGB` float32 ndarray in [0, 1]
- Common output: `Image(ydata, cbdata, crdata, metadata_dict)` — a simple dataclass

NV12 layout: Y plane (W×H) followed by interleaved UV plane (W×H/2). Chroma is upsampled to full resolution at read time so all downstream code works on same-size planes.

### 3.2 Layer 2 — Preprocessing (`preprocess/`)

All modules receive a shared `PreprocessedImage` containing:
- `y`, `cb`, `cr` — float32 ndarrays, full resolution
- `roi` — dict of boolean masks: `center`, `edge_{top,bot,left,right}`, `corner_{tl,tr,bl,br}`
- `detail_mask` — pixels where local variance exceeds threshold (texture zone)
- `flat_mask` — pixels where local variance is below threshold
- `edge_mask` — Canny edge binary mask
- `gradient_mag` — Sobel gradient magnitude on Y channel

ROI geometry: center = middle 25% area; edges = 15% strips along each side; corners = 10% squares at each corner.

### 3.3 Layer 3 — Metric Modules (`metrics/`)

Each module is a function with this signature:

```python
def metric(image: PreprocessedImage) -> MetricResult:
    ...

@dataclass
class MetricResult:
    name: str               # e.g. "sharpness"
    global_score: float     # 0–100
    heatmap: ndarray        # same size as image, pixel-level quality
    region_scores: dict     # {region_name: score}
    diagnosis: str          # human-readable, spatially specific
    metadata: dict          # raw intermediate values (MTF50, SNR, DeltaE...)
```

**11 modules and their core algorithms:**

| # | Module | Algorithm (library) | Key intermediates |
|---|--------|---------------------|-------------------|
| 1 | Exposure | Histogram analysis on Y (numpy), highlight/shadow clip % | mean_Y, clip_ratio |
| 2 | Brightness | ITU-R BT.709 perceived brightness (numpy), nonlinear mapping to 100 | perceived_brightness |
| 3 | Contrast | Local RMS contrast + global Michelson (numpy) | rms_map, michelson |
| 4 | Color Accuracy | Mean chroma deviation from gray-world neutral (numpy), saturation check | delta_Cb, delta_Cr |
| 5 | White Balance | Gray-world + white-patch estimation (numpy), Cb/Cr neutrality | estimated_illuminant_K |
| 6 | Sharpness | Edge profile slope (opencv Sobel + scipy sigmoid fit), MTF50 estimation via slanted-edge (scipy) | mtf50_per_roi, edge_width_px |
| 7 | Noise | ISO 15739 visual noise on Y + chroma noise on CbCr (skimage noise estimation) | snr_db, visual_noise, chroma_noise |
| 8 | Dynamic Range | Highlight/shoulder clipping SNR, shadow noise floor (numpy) | dr_stops, headroom_stops |
| 9 | Texture Preservation | Local variance in detail_mask vs flat_mask ratio (skimage texture) | texture_to_flat_variance_ratio |
| 10 | Uniformity (Shading) | Corner-to-center luminance/color ratio (numpy per-ROI stats) | corner_vs_center_luma_ratio |
| 11 | Color Fringing | Purple/magenta detection near high-contrast edges in CbCr space (opencv) | fringe_pixel_ratio |

### 3.4 Layer 4 — Aggregation (`aggregate/`)

- **Single device**: weighted sum → `total_score = sum(w[i] * score[i])`, weights TBD (equal default)
- **Dual device**: per-metric `delta = score_test - score_ref`
- **Diagnosis filter**: metrics with score < 60 get flagged, diagnosis text includes ROI-specific location

### 3.5 Layer 5 — Report (`report/`)

- `print_report(results)` — formatted table in console
- `render_report(results, output_path)` — saves `report.png` containing:
  - Radar chart (matplotlib polar) with 11 axes
  - 3×4 grid of heatmaps (matplotlib imshow) overlaying original Y channel
  - Diagnosis text block with flagged issues highlighted

## 4. Project File Structure

```
F:\AI\AItuning\
├── scorer/
│   ├── __init__.py
│   ├── pipeline.py          # orchestrates layers 1→5
│   ├── io/
│   │   ├── __init__.py
│   │   ├── nv12.py          # NV12 reader + chroma upsampling
│   │   ├── png_reader.py    # PNG → float32 RGB
│   │   └── image_model.py   # Image dataclass
│   ├── preprocess/
│   │   ├── __init__.py
│   │   ├── color_space.py   # RGB↔YCbCr, YUV↔RGB
│   │   ├── roi.py           # ROI mask generation
│   │   └── feature_maps.py  # edge/texture/flat masks, gradients
│   ├── metrics/
│   │   ├── __init__.py      # MetricResult dataclass, registry
│   │   ├── exposure.py
│   │   ├── brightness.py
│   │   ├── contrast.py
│   │   ├── color_accuracy.py
│   │   ├── white_balance.py
│   │   ├── sharpness.py
│   │   ├── noise.py
│   │   ├── dynamic_range.py
│   │   ├── texture.py
│   │   ├── uniformity.py
│   │   └── fringing.py
│   ├── aggregate.py          # weighted total, comparison delta
│   └── report/
│       ├── __init__.py
│       ├── console.py        # text table output
│       └── render.py         # matplotlib charts + heatmaps → PNG
├── run.py                    # CLI entry point
├── tests/
│   ├── test_io.py
│   ├── test_preprocess.py
│   └── test_metrics.py
└── docs/superpowers/specs/2026-05-12-image-quality-scoring-design.md
```

## 5. CLI Interface

```
python run.py <image_path> [--width W] [--height H] [--ref <ref_image>]
```

NV12 requires `--width` and `--height`; PNG auto-detects. Reference image is optional.

## 6. Error Handling

- Missing image file → exit with message
- NV12 without width/height → exit with message
- Unsupported format → exit with message
- scipy/skimage import failure → exit with pip install instructions

No recovery from bad inputs; fail fast with clear messages.

## 7. Testing Strategy

### Unit tests
- Each IO reader: roundtrip a known synthetic image (e.g., 64×64 gradient)
- Each metric: known-input → expected-output (e.g., pure gray returns score 100 for noise, uniform white returns score 100 for uniformity)
- ROI masks: verify coverage ratios
- Color space: verify BT.709 conversion against known reference values

### Integration test
- Synthesize a "perfect" image (uniform gray) → all scores near 100
- Synthesize a "degraded" image (add noise, blur, tint) → scores drop
- Compare relative ordering: perfect > degraded in every dimension

### Manual validation
- 3 real camera images covering good/moderate/poor quality
- Run scoring, inspect report, verify rankings match human judgment

## 8. Implementation Order

1. `io/` — image reading (can't test anything without it)
2. `preprocess/` — ROI + feature maps (required by all metrics)
3. `metrics/exposure.py` + `brightness.py` + `contrast.py` — simplest (histogram + stats)
4. `metrics/sharpness.py` + `noise.py` — core perceptual metrics
5. Remaining metrics
6. `aggregate.py` + `report/` — ties it all together
7. `run.py` — CLI entry point

## 9. Dependencies

All already installed in `F:\AI\imglab\.venv`:
- numpy 2.4.4, scipy 1.17.1, opencv-python 4.13.0.92
- matplotlib 3.10.8, scikit-image 0.26.0, pillow 12.2.0

No additional packages needed.
