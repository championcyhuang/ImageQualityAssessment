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
        assert len(output["results"]) == 13
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
        assert len(output["results"]) == 13
    finally:
        os.unlink(tmp)
