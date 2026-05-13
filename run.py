#!/usr/bin/env python3
"""Image Quality Scoring System — CLI entry point.

Usage:
    python run.py <image_path> [--width W] [--height H] [--ref <ref_image>]
"""

import argparse
import sys

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from scorer.pipeline import run_pipeline
from scorer.report.console import print_report
from scorer.report.render import render_report
from scorer.report.html_render import render_html


def main():
    parser = argparse.ArgumentParser(description="Image Quality Scoring System")
    parser.add_argument("image", help="Path to input image (.nv12 or .png)")
    parser.add_argument("--width", type=int, help="Image width (required for .nv12)")
    parser.add_argument("--height", type=int, help="Image height (required for .nv12)")
    parser.add_argument("--ref", help="Path to reference device image (optional)")
    parser.add_argument("--output", "-o", default="report.html",
                        help="Output report path (default: report.html, use .png for image)")
    args = parser.parse_args()

    try:
        output = run_pipeline(args.image, args.width, args.height, args.ref)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print_report(output["results"], output["total_score"])

    if output["ref_results"]:
        print("\n" + "=" * 70)
        print("  Ref vs Test Score Deltas")
        print("=" * 70)
        for d in output["deltas"]:
            direction = ">" if d["delta"] > 0 else ("<" if d["delta"] < 0 else "=")
            print(f"  {d['metric']:<20} Test:{d['test_score']:>5.1f}  "
                  f"Ref:{d['ref_score']:>5.1f}  Delta:{d['delta']:>+6.1f} {direction}")
        print(f"\n  Test Total: {output['total_score']:.1f}  |  "
              f"Ref Total: {output['ref_total_score']:.1f}")

    # Render visual report
    ext = args.output.rsplit(".", 1)[-1].lower() if "." in args.output else "html"
    if ext == "png":
        render_report(output["results"], output["total_score"],
                      output["image_y"], args.output)
    else:
        render_html(output["results"], output["total_score"],
                    output["image_y"], args.output,
                    ref_results=output.get("ref_results"),
                    ref_total_score=output.get("ref_total_score"))
    print(f"\n[Report] Saved to: {args.output}")


if __name__ == "__main__":
    main()
