import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from ..metrics import MetricResult
from ..aggregate import compute_total_score

# Register CJK font and force it as default for all text
_CJK_FONT_PATH = "C:/Windows/Fonts/msyh.ttc"
try:
    font_manager.fontManager.addfont(_CJK_FONT_PATH)
    _prop = font_manager.FontProperties(fname=_CJK_FONT_PATH)
    _cjk_name = _prop.get_name()
    matplotlib.rcParams["font.family"] = _cjk_name
except Exception:
    pass


def render_report(results: list[MetricResult], total_score: float,
                  image_y: np.ndarray | None, output_path: str):
    """Render visual quality report to PNG.

    Layout: radar chart (left) + 3x4 heatmap grid (right) + diagnosis text at bottom.
    """
    fig = plt.figure(figsize=(20, 18))

    # Left: Radar chart
    ax_radar = fig.add_axes([0.02, 0.55, 0.28, 0.40], polar=True)
    _draw_radar(ax_radar, results, total_score)

    # Right: Heatmap grid (3 rows x 4 cols)
    _draw_heatmap_grid(fig, results, image_y)

    # Bottom: Diagnosis text
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
    """Draw 3x4 grid of heatmaps."""
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

    # Hide unused subplot cell (grid is 3x4=12, only 11 metrics)
    ax = fig.add_axes([0.35 + 3 * 0.16, 0.55 - 2 * 0.30, 0.13, 0.12])
    ax.axis("off")


def _draw_diagnosis_text(ax, results, total_score):
    """Draw text summary of issues flagged."""
    lines = []
    lines.append(f"综合总分: {total_score:.1f}/100")
    for r in results:
        if r.global_score < 60:
            lines.append(f"[!!] {r.name}: {r.global_score:.1f} — {r.diagnosis}")

    if len(lines) == 1:
        lines.append("所有指标均在正常范围，图像质量合格。")

    text = "\n".join(lines)
    ax.text(0.01, 0.5, text, fontsize=10,
            verticalalignment="center", transform=ax.transAxes)
