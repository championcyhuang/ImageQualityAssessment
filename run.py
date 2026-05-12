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
