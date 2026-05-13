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
        dict with: results, total_score, ref_results, ref_total_score, deltas, image_y
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
        "image_rgb": img.rgb,
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
    elif ext in (".png", ".jpg", ".jpeg"):
        return read_png(path)
    else:
        raise ValueError(f"Unsupported format: {ext}. Use .nv12, .png, .jpg, .jpeg")
