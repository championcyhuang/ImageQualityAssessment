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

# Algorithm descriptions for each metric
_ALGO_DESCRIPTIONS = {
    "exposure":            "Y通道直方图分析，目标均值0.35-0.65，惩罚高光裁切(Y>0.95)和暗部裁切(Y<0.05)",
    "brightness":          "ITU-R BT.709感知亮度，Gamma 2.2近似，目标感知亮度~0.72(18%灰)，偏离扣分",
    "contrast":            "局部RMS对比度(8x8块std/mean) + 全局Michelson对比度(p95-p5)/(p95+p5)，各占50%",
    "color_accuracy":      "灰世界假设 — Cb/Cr色度通道偏离中性灰(0,0)的程度，delta_C = sqrt(Cb^2+Cr^2)",
    "white_balance":       "灰世界(70%)+白点(30%)双假设法，分析最亮像素(95%百分位)的色度偏移，估算光源色温",
    "sharpness":           "Laplacian方差(经典对焦测度)，cv2.Laplacian + 梯度剖面散度估算边缘宽度(简化MTF法)",
    "noise":               "ISO 15739视觉噪声法 — 平坦区SNR估计(20*log10(signal/noise))，亮度噪声+色度噪声综合评分",
    "dynamic_range":       "直方图有效像素P1-P99范围，DR = log2(p99/p01) stops，惩罚高光/暗部裁切比例",
    "texture_preservation": "平坦区与细节区方差比值法，检测降噪是否过度抹除纹理，ratio = detail_var / flat_var",
    "uniformity":          "四角vs中心亮度比(镜头Shading评估)，角落亮度/中心亮度，取最差角落比值评分",
    "fringing":            "高对比度边缘邻域紫边/青边检测，Cr>0.05&Cb<-0.03(紫边)/Cr<-0.03&Cb>0.03(青边)像素占比",
    "saturation":          "YCbCr色度幅值分析，chroma_mag = sqrt(Cb^2+Cr^2)，目标范围0.05-0.15，惩罚欠饱和和过饱和",
}


def render_report(results: list[MetricResult], total_score: float,
                  image_y: np.ndarray | None, output_path: str):
    """Render visual quality report to PNG.

    Layout (top to bottom):
      Row 1: original image (left) + radar chart (right)
      Row 2: 3x4 heatmap grid
      Row 3: diagnosis text
      Row 4: algorithm descriptions (compact table)
    """
    fig = plt.figure(figsize=(26, 32))

    # Row 1: Original image (top-left)
    if image_y is not None:
        ax_img = fig.add_axes([0.02, 0.78, 0.18, 0.16])
        ax_img.imshow(image_y, cmap="gray", aspect="auto")
        ax_img.set_title("Y", fontsize=8)
        ax_img.axis("off")

    # Row 1: Radar chart (right of image)
    ax_radar = fig.add_axes([0.22, 0.72, 0.28, 0.25], polar=True)
    _draw_radar(ax_radar, results, total_score)

    # Row 2: Heatmap grid (3 rows x 4 cols), properly spaced on right side
    _draw_heatmap_grid(fig, results)

    # Row 3: Diagnosis text
    ax_diag = fig.add_axes([0.05, 0.08, 0.90, 0.07])
    ax_diag.axis("off")
    _draw_diagnosis_text(ax_diag, results, total_score)

    # Row 4: Algorithm descriptions
    ax_algo = fig.add_axes([0.03, 0.005, 0.94, 0.07])
    ax_algo.axis("off")
    _draw_algorithm_table(ax_algo, results)

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
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=6)
    ax.fill(angles_plot, scores_plot, alpha=0.25, color="#2196F3")
    ax.plot(angles_plot, scores_plot, color="#2196F3", linewidth=2)
    ax.set_title(f"Total: {total_score:.1f}/100", fontsize=11, fontweight="bold", pad=18)


def _draw_heatmap_grid(fig, results):
    """Draw 3x4 grid of heatmaps with titles separated from images."""
    n_cols = 4
    cell_w = 0.105
    cell_h = 0.12
    x_start = 0.54
    y_bases = [0.60, 0.43, 0.26]

    for idx, r in enumerate(results):
        col = idx % n_cols
        row = idx // n_cols
        x = x_start + col * (cell_w + 0.01)
        y = y_bases[row]

        # Main axes for heatmap image only
        ax_img = fig.add_axes([x, y, cell_w, cell_h])
        hm = r.heatmap
        if hm is not None and hm.size > 1:
            ax_img.imshow(hm, cmap="hot", interpolation="nearest", aspect="auto")
        else:
            ax_img.text(0.5, 0.5, f"{r.global_score:.1f}", ha="center", va="center",
                        fontsize=12, transform=ax_img.transAxes)
        ax_img.axis("off")

        # Title above the heatmap (separate axes to prevent overlap)
        ax_title = fig.add_axes([x, y + cell_h, cell_w, 0.03])
        ax_title.axis("off")
        ax_title.text(0.5, 0.3, f"{r.name}\n{r.global_score:.1f}",
                      ha="center", va="bottom", fontsize=6.5,
                      transform=ax_title.transAxes)

    # 3x4 grid fits exactly 12 metrics, no unused cells


def _draw_diagnosis_text(ax, results, total_score):
    """Draw text summary of issues flagged."""
    lines = [f"Total Score: {total_score:.1f}/100"]
    for r in results:
        if r.global_score < 60:
            lines.append(f"[!!] {r.name}: {r.global_score:.1f}  {r.diagnosis}")

    if len(lines) == 1:
        lines.append("All metrics within normal range.")

    text = " | ".join(lines)
    ax.text(0.01, 0.5, text, fontsize=9,
            verticalalignment="center", transform=ax.transAxes)


def _draw_algorithm_table(ax, results):
    """Draw compact algorithm descriptions in a two-column table."""
    desc_map = {}
    for r in results:
        desc_map[r.name] = _ALGO_DESCRIPTIONS.get(r.name, "")

    lines = []
    # Two-column layout: split into left and right columns
    mid = (len(results) + 1) // 2
    left = results[:mid]
    right = results[mid:]

    max_name_len = max(len(r.name) for r in results)

    for i in range(mid):
        left_line = ""
        if i < len(left):
            ln = left[i]
            left_line = f"{ln.name:<20} | {desc_map.get(ln.name, '')}"
        right_line = ""
        if i < len(right):
            rn = right[i]
            right_line = f"{rn.name:<20} | {desc_map.get(rn.name, '')}"

        combined = left_line
        if right_line:
            combined += "    " + right_line
        lines.append(combined)

    text = "\n".join(lines)
    ax.text(0.01, 0.5, text, fontsize=6.5, verticalalignment="center",
            transform=ax.transAxes)
