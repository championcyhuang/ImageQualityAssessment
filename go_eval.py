#!/usr/bin/env python3
"""Image Quality Evaluation Orchestrator.

Scans input directory, evaluates each image independently,
and when 2+ images are found, runs comparison (first as reference).
Generates a single multi-tab HTML report.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from scorer.pipeline import run_pipeline
from scorer.report.console import print_report
from scorer.report.html_render import render_multi_report

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".nv12"}


def find_images(input_dir: str) -> list[Path]:
    """Find unique supported image files, sorted by name."""
    input_path = Path(input_dir)
    if not input_path.exists():
        return []
    seen = set()
    images = []
    for ext in SUPPORTED_EXTENSIONS:
        for p in input_path.iterdir():
            if p.suffix.lower() == ext and p.name not in seen:
                seen.add(p.name)
                images.append(p)
    return sorted(images, key=lambda p: p.name)


def main():
    parser = argparse.ArgumentParser(description="Image Quality Evaluation Orchestrator")
    _here = Path(__file__).parent
    parser.add_argument("--input", "-i",
                        default=str(_here / "需要评估的图片"),
                        help="Input directory with images to evaluate")
    parser.add_argument("--output", "-o",
                        default=str(_here / "评估报告"),
                        help="Output directory for reports")
    args = parser.parse_args()

    images = find_images(args.input)
    if not images:
        print(f"No supported images (.png, .jpg, .jpeg, .nv12) found in: {args.input}")
        sys.exit(0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Image Quality Evaluation")
    print(f"  Input:  {args.input}")
    print(f"  Output: {output_dir}")
    print(f"  Found:  {len(images)} image(s)")
    for img in images:
        print(f"    - {img.name}")
    print(f"{'='*70}")

    # Evaluate every image independently
    all_results = []
    for img in images:
        print(f"\n--- Evaluating: {img.name} ---")
        try:
            out = run_pipeline(str(img))
        except Exception as e:
            print(f"  SKIP: {e}")
            continue
        print_report(out["results"], out["total_score"])
        all_results.append({
            "name": img.name,
            "results": out["results"],
            "total_score": out["total_score"],
            "image_y": out["image_y"],
            "image_rgb": out.get("image_rgb"),
        })

    # Comparison (if 2+ images): first as reference
    comparison_pairs = []
    if len(images) >= 2:
        ref_name = images[0].name
        for img in images[1:]:
            print(f"\n--- Compare: {img.name} vs {ref_name} ---")
            out = run_pipeline(str(img), ref_path=str(images[0]))
            print_report(out["results"], out["total_score"])
            if out["ref_results"]:
                print(f"  Ref: {out['ref_total_score']:.1f}  |  "
                      f"Test: {out['total_score']:.1f}  |  "
                      f"Delta: {out['total_score'] - out['ref_total_score']:+.1f}")
            comparison_pairs.append({
                "test_name": img.name,
                "ref_name": ref_name,
                "results": out["results"],
                "total_score": out["total_score"],
                "ref_results": out["ref_results"],
                "ref_total_score": out["ref_total_score"],
                "image_y": out["image_y"],
                "image_rgb": out.get("image_rgb"),
            })

    # Render single multi-tab HTML report
    report_path = output_dir / "quality_report.html"
    render_multi_report(all_results, comparison_pairs, str(report_path))
    print(f"\n  Report saved: {report_path}")

    print(f"\n{'='*70}")
    print(f"  Evaluation complete. Reports: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
