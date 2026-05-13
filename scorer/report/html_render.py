"""HTML report renderer — multi-tab, Champion-inspired, self-contained.

Tab switching uses minimal inline JS for robustness across tab counts.
"""

import base64
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from ..metrics import MetricResult

_CJK_FONT_PATH = "C:/Windows/Fonts/msyh.ttc"
try:
    font_manager.fontManager.addfont(_CJK_FONT_PATH)
    matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=_CJK_FONT_PATH).get_name()
    matplotlib.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

_ALGO_DESCRIPTIONS = {
    "exposure": "Y通道直方图分析，目标均值0.35-0.65，惩罚高光裁切(Y>0.95)和暗部裁切(Y<0.05)",
    "brightness": "ITU-R BT.709感知亮度，Gamma 2.2近似，目标感知亮度~0.72(18%灰)，偏离扣分",
    "contrast": "局部RMS对比度(8x8块std/mean) + 全局Michelson对比度(p95-p5)/(p95+p5)，各占50%",
    "color_accuracy": "灰世界假设 — Cb/Cr色度通道偏离中性灰(0,0)的程度，delta_C = sqrt(Cb^2+Cr^2)",
    "white_balance": "灰世界(70%)+白点(30%)双假设法，分析最亮像素(95%百分位)的色度偏移，估算光源色温",
    "sharpness": "Laplacian方差(经典对焦测度)，cv2.Laplacian + 梯度剖面散度估算边缘宽度(简化MTF法)",
    "noise": "ISO 15739视觉噪声法 — 平坦区SNR估计，亮度噪声+色度噪声综合评分",
    "dynamic_range": "直方图有效像素P1-P99范围，DR = log2(p99/p01) stops，惩罚高光/暗部裁切比例",
    "texture_preservation": "平坦区与细节区方差比值法，检测降噪是否过度抹除纹理",
    "uniformity": "四角vs中心亮度比(镜头Shading评估)，取最差角落比值评分",
    "fringing": "高对比度边缘邻域紫边/青边检测，Cr>0.05&Cb<-0.03/Cr<-0.03&Cb>0.03",
    "saturation": "YCbCr色度幅值分析，chroma_mag = sqrt(Cb²+Cr²)，目标范围0.05-0.15",
    "distortion": "轮廓直线度分析——检测长边缘、拟合直线、测量点线平均偏离，偏离越大畸变越严重",
}

_METRIC_CN = {
    "exposure": "曝光", "brightness": "亮度", "contrast": "对比度",
    "color_accuracy": "色彩准确度", "white_balance": "白平衡",
    "sharpness": "清晰度", "noise": "噪声", "dynamic_range": "动态范围",
    "texture_preservation": "纹理保留", "uniformity": "均匀性",
    "fringing": "色散", "saturation": "饱和度",
    "distortion": "畸变",
}

_CSS = r"""
:root {
  --bg:#f5efe5; --bg2:#fbf7f1; --card:rgba(255,251,246,0.94);
  --line:rgba(22,49,58,0.08); --ink:#16313a; --muted:#5e7178;
  --green:#24584c; --green2:#3d7b6b; --orange:#cf7235; --orange2:#e59a58;
  --red:#c44a3f; --blue:#3a6b8c;
  --mono:"Cascadia Code","JetBrains Mono","Consolas",monospace;
  --sans:"Noto Sans SC","Microsoft YaHei","PingFang SC","Segoe UI",sans-serif;
  --radius-xl:28px; --radius-lg:20px; --radius-md:14px; --radius-sm:10px;
  --shadow:0 16px 40px rgba(20,35,42,0.10); --shadow-sm:0 4px 12px rgba(20,35,42,0.06);
}
* { margin:0; padding:0; box-sizing:border-box; }
html { scroll-behavior:smooth; }
body {
  color:var(--ink); font-family:var(--sans); line-height:1.6;
  background:
    radial-gradient(circle at 10% 10%, rgba(229,154,88,0.10), transparent 22%),
    radial-gradient(circle at 90% 14%, rgba(61,123,107,0.10), transparent 20%),
    linear-gradient(180deg, #efe7da 0%, #f8f4ed 48%, #f3ece1 100%);
}
.shell { width:min(1480px, calc(100% - 32px)); margin:0 auto; }

/* Topbar */
.topbar { position:sticky; top:0; z-index:30; border-bottom:1px solid rgba(22,49,58,0.07); background:rgba(251,247,241,0.86); backdrop-filter:blur(14px); }
.topbar-inner { display:flex; align-items:center; justify-content:space-between; gap:14px; padding:14px 0; }
.brand { display:flex; align-items:center; gap:10px; }
.brand-mark { width:42px; height:42px; border-radius:12px; display:grid; place-items:center; color:#fff8ef; font-weight:900; font-size:18px; background:linear-gradient(140deg, var(--green) 0%, var(--orange) 100%); box-shadow:0 12px 28px rgba(35,88,77,0.22); }
.brand-title { font-size:1rem; font-weight:900; }
.brand-sub { font-size:0.82rem; color:var(--muted); }

/* Tab navigation */
.tab-nav { display:flex; gap:6px; margin:18px 0 16px; flex-wrap:wrap; }
.tab-radio { display:none; }
.tab-label {
  display:inline-flex; align-items:center; gap:6px;
  padding:10px 20px; border-radius:999px; cursor:pointer;
  font-size:0.9rem; font-weight:700; color:var(--muted);
  background:rgba(255,255,255,0.6); border:1px solid var(--line);
  transition:0.2s ease;
}
.tab-label:hover { background:rgba(255,255,255,0.9); color:var(--ink); }
.tab-label .tab-badge {
  display:inline-flex; align-items:center; justify-content:center;
  min-width:28px; height:22px; border-radius:999px; padding:0 8px;
  font-size:0.75rem; font-weight:800; color:#fff;
}
.tab-radio:checked + .tab-label {
  color:#fff8ef; border-color:transparent;
  background:linear-gradient(140deg, var(--green) 0%, var(--orange) 100%);
  box-shadow:0 10px 22px rgba(35,88,77,0.18);
}
.tab-radio:checked + .tab-label .tab-badge { background:rgba(255,255,255,0.25); }

/* Tab panels */
.tab-panel { display:none; }

/* Hero */
.hero { padding:20px 0 14px; }
.hero-panel {
  overflow:hidden; position:relative; border-radius:var(--radius-xl);
  padding:28px 32px; color:#fff8f0;
  background:linear-gradient(140deg, rgba(18,44,52,0.97) 0%, rgba(35,88,77,0.93) 46%, rgba(207,114,53,0.90) 100%);
  box-shadow:var(--shadow);
}
.hero-panel::before { content:""; position:absolute; width:280px; height:280px; border-radius:50%; right:-50px; top:-110px; background:radial-gradient(circle, rgba(255,238,212,0.18), transparent 62%); }
.hero-panel * { position:relative; z-index:1; }
.hero-eyebrow { display:inline-flex; align-items:center; gap:8px; padding:7px 12px; border-radius:999px; background:rgba(255,255,255,0.12); color:#fff0dd; font-size:0.82rem; font-weight:800; letter-spacing:0.06em; text-transform:uppercase; }
.hero-panel h1 { margin:14px 0 8px; font-size:clamp(2rem,3.5vw,2.6rem); line-height:1.0; letter-spacing:-0.04em; }
.hero-score { display:inline-flex; align-items:baseline; gap:6px; margin-top:4px; }
.hero-score .val { font-size:3.2rem; font-weight:900; line-height:1; }
.hero-score .unit { font-size:1rem; opacity:0.75; }
.hero-meta { margin-top:10px; font-size:0.9rem; opacity:0.8; }

/* Section */
.section-hd { font-size:1.2rem; font-weight:800; margin:24px 0 14px; color:var(--ink); }

/* Row: source + radar */
.row2 { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:18px; }
.panel { border-radius:var(--radius-lg); background:var(--card); border:1px solid var(--line); box-shadow:var(--shadow-sm); overflow:hidden; margin-bottom:14px; }
.panel-inner { padding:18px 20px; text-align:center; }
.panel-inner img { max-width:100%; border-radius:var(--radius-sm); }

/* Summary dashboard */
.summary-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(280px, 1fr)); gap:14px; margin-bottom:18px; }
.summary-card {
  border-radius:var(--radius-lg); overflow:hidden; background:var(--card);
  border:1px solid var(--line); box-shadow:var(--shadow-sm); cursor:pointer;
  transition:0.2s ease; text-decoration:none; color:inherit; display:block;
}
.summary-card:hover { transform:translateY(-2px); box-shadow:var(--shadow); }
.summary-top { padding:20px 20px 14px; }
.summary-name { font-size:0.85rem; color:var(--muted); margin-bottom:2px; }
.summary-score { font-size:2.4rem; font-weight:900; }
.summary-bar-wrap { margin:0 20px 12px; }
.summary-bar { height:6px; border-radius:3px; background:rgba(22,49,58,0.06); overflow:hidden; }
.summary-bar-fill { height:100%; border-radius:3px; }
.summary-stats { display:grid; grid-template-columns:1fr 1fr; gap:6px; padding:12px 20px; border-top:1px solid var(--line); }
.summary-stat { text-align:center; }
.summary-stat .sv { font-size:1.1rem; font-weight:800; }
.summary-stat .sl { font-size:0.72rem; color:var(--muted); }

/* Metrics grid */
.metrics-grid { display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:12px; margin-bottom:18px; }
.metric-card {
  border-radius:var(--radius-lg); overflow:hidden; background:var(--card);
  border:1px solid var(--line); box-shadow:var(--shadow-sm);
}
.metric-head { display:flex; align-items:center; justify-content:space-between; padding:12px 14px 8px; }
.metric-head .mname { font-size:0.85rem; font-weight:800; }
.metric-head .mscore { font-size:1.35rem; font-weight:900; }
.metric-bar-wrap { margin:0 14px 6px; }
.metric-bar { height:5px; border-radius:3px; background:rgba(22,49,58,0.06); overflow:hidden; }
.metric-bar-fill { height:100%; border-radius:3px; }
.metric-img { padding:0 8px 6px; }
.metric-img img { width:100%; border-radius:var(--radius-sm); display:block; }
.metric-diag { padding:6px 14px 10px; font-size:0.76rem; color:var(--muted); line-height:1.5; border-top:1px solid var(--line); }

/* Compare table */
.compare-table { width:100%; border-collapse:collapse; font-size:0.88rem; }
.compare-table th { text-align:left; padding:10px 14px; color:var(--muted); font-weight:700; border-bottom:2px solid var(--line); font-size:0.76rem; text-transform:uppercase; }
.compare-table td { padding:10px 14px; border-bottom:1px solid var(--line); }
.compare-table tr:last-child td { border-bottom:none; }
.delta-up { color:var(--green); font-weight:800; }
.delta-down { color:var(--red); font-weight:800; }

/* Diagnosis */
.diag-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:10px; }
.diag-card { border-radius:var(--radius-md); padding:14px 16px; background:var(--card); border:1px solid var(--line); border-left:4px solid var(--red); }
.diag-card .dname { font-weight:800; color:var(--red); margin-bottom:4px; }
.diag-card .dtext { font-size:0.84rem; color:var(--muted); line-height:1.5; }

/* Algo */
.algo-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:8px; margin-bottom:14px; }
.algo-item { border-radius:var(--radius-sm); padding:10px 14px; background:var(--card); border:1px solid var(--line); font-size:0.84rem; }
.algo-item .aname { color:var(--green); font-weight:800; }

.footer { text-align:center; color:var(--muted); font-size:0.82rem; margin:28px 0 20px; padding-top:16px; border-top:1px solid var(--line); }
"""


def _fig_to_b64(fig: plt.Figure, dpi: int = 120) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none",
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _make_radar_figure(results: list[MetricResult], total_score: float) -> str:
    names = [_METRIC_CN.get(r.name, r.name) for r in results]
    scores = [r.global_score for r in results]
    N = len(names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    scores_plot, angles_plot = scores + [scores[0]], angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(5.2, 4.8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#fbf7f1")
    ax.set_facecolor("#fbf7f1")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(names, fontsize=7.5, color="#16313a")
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=6, color="#5e7178")
    ax.fill(angles_plot, scores_plot, alpha=0.20, color="#24584c")
    ax.plot(angles_plot, scores_plot, color="#24584c", linewidth=2.5)
    ax.set_title(f"{total_score:.1f} / 100", fontsize=14, fontweight="900", color="#16313a", pad=20)
    ax.tick_params(colors="#5e7178")
    for spine in ax.spines.values():
        spine.set_edgecolor("#d0ccc4")
    return _fig_to_b64(fig, dpi=130)


def _make_simulation_figure(image_y: np.ndarray, heatmap: np.ndarray, score: float) -> str:
    """Y-channel base + heatmap semi-transparent color overlay."""
    if image_y is None or image_y.size <= 1:
        return ""
    h, w = image_y.shape
    p2, p98 = np.percentile(image_y, 2), np.percentile(image_y, 98)
    base = np.clip((image_y - p2) / max(p98 - p2, 1e-8), 0, 1)
    base_rgb = np.stack([base, base, base], axis=-1)

    if heatmap is not None and heatmap.size > 1:
        hp2, hp98 = np.percentile(heatmap, 2), np.percentile(heatmap, 98)
        hm_norm = np.clip((heatmap - hp2) / max(hp98 - hp2, 1e-8), 0, 1) if hp98 > hp2 else np.zeros_like(heatmap)
    else:
        hm_norm = np.zeros_like(image_y)

    if score >= 70:
        overlay_color = np.array([61, 123, 107]) / 255.
    elif score >= 50:
        overlay_color = np.array([229, 154, 88]) / 255.
    else:
        overlay_color = np.array([196, 74, 63]) / 255.

    overlay = hm_norm[:, :, np.newaxis] * overlay_color * 0.7
    combined = np.clip(base_rgb * (1 - hm_norm[:, :, np.newaxis] * 0.4) + overlay, 0, 1)

    fig, ax = plt.subplots(figsize=(3.6, 2.6), dpi=120)
    ax.imshow(combined, aspect="auto")
    ax.axis("off")
    fig.patch.set_facecolor("#fbf7f1")
    fig.subplots_adjust(0, 0, 1, 1)
    return _fig_to_b64(fig, dpi=120)


def _make_source_figure(image_rgb: np.ndarray | None, image_y: np.ndarray | None) -> str:
    """Show original RGB or Y-channel preview."""
    if image_rgb is not None and image_rgb.size > 1:
        fig, ax = plt.subplots(figsize=(4, 3), dpi=120)
        ax.imshow(image_rgb)
        ax.set_title("Original Image", fontsize=9, color="#16313a", fontweight="bold", pad=4)
        ax.axis("off")
        fig.patch.set_facecolor("#fbf7f1")
        fig.subplots_adjust(0, 0, 1, 0.92)
        return _fig_to_b64(fig, dpi=120)
    elif image_y is not None:
        arr = image_y.copy()
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            p2, p98 = np.percentile(arr, 2), np.percentile(arr, 98)
            arr = np.clip((arr - p2) / max(p98 - p2, 1e-8) * 255, 0, 255).astype(np.uint8)
        fig, ax = plt.subplots(figsize=(4, 3), dpi=120)
        ax.imshow(arr, cmap="gray", aspect="auto")
        ax.set_title("Source (Y Channel)", fontsize=9, color="#16313a", fontweight="bold", pad=4)
        ax.axis("off")
        fig.patch.set_facecolor("#fbf7f1")
        fig.subplots_adjust(0, 0, 1, 0.92)
        return _fig_to_b64(fig, dpi=120)
    return ""


def _sc(score: float) -> str:
    return "var(--green)" if score >= 70 else "var(--orange)" if score >= 50 else "var(--red)"


def _sc_bg(score: float) -> str:
    return "#24584c" if score >= 70 else "#cf7235" if score >= 50 else "#c44a3f"


def _sc_tab(score: float) -> str:
    return "#24584c" if score >= 70 else "#cf7235" if score >= 50 else "#c44a3f"


def render_multi_report(all_results: list[dict], comparison_pairs: list[dict], output_path: str):
    """Render multi-tab HTML report for one or more evaluated images.

    Args:
        all_results: list of {name, results, total_score, image_y, image_rgb}
        comparison_pairs: list of {test_name, ref_name, results, total_score,
                          ref_results, ref_total_score, image_y, image_rgb}
    """
    p = []
    p.append("<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>")
    p.append("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    p.append("<title>Image Quality Report</title>")
    p.append(f"<style>{_CSS}</style></head><body>")

    # ── Topbar ──
    p.append("<div class='topbar'><div class='shell topbar-inner'>")
    p.append("<div class='brand'><div class='brand-mark'>IQ</div>")
    p.append("<div><div class='brand-title'>Image Quality Scoring</div>")
    p.append(f"<div class='brand-sub'>{len(all_results)} image(s) evaluated</div></div></div>")
    p.append(f"<div style='font-size:0.84rem;color:var(--muted)'>12 metrics &middot; AItuning</div>")
    p.append("</div></div>")

    # ── Tab navigation ──
    p.append("<div class='shell'>")
    p.append("<div class='tab-nav'>")

    # Summary tab (always first)
    p.append("<input type='radio' name='tab' id='tab-summary' class='tab-radio' checked>")
    p.append("<label class='tab-label' for='tab-summary'>Summary</label>")

    # One tab per image
    for i, r in enumerate(all_results):
        tid = f"tab-img{i}"
        badge_color = _sc_tab(r["total_score"])
        p.append(f"<input type='radio' name='tab' id='{tid}' class='tab-radio'>")
        p.append(f"<label class='tab-label' for='{tid}'>")
        p.append(f"<span class='tab-badge' style='background:{badge_color}'>{r['total_score']:.0f}</span>")
        p.append(f"{r['name']}</label>")

    # Comparison tab (if data available)
    if comparison_pairs:
        p.append("<input type='radio' name='tab' id='tab-compare' class='tab-radio'>")
        p.append("<label class='tab-label' for='tab-compare'>Comparison</label>")

    p.append("</div>")

    # ── Tab content ──
    p.append("<div class='tab-content'>")

    # --- Summary Panel ---
    p.append("<div id='panel-summary' class='tab-panel'>")
    p.append("<div class='hero'><div class='hero-panel'>")
    p.append("<div class='hero-eyebrow'>EVALUATION SUMMARY</div>")
    p.append(f"<h1>{len(all_results)} Image(s)</h1>")
    p.append("<div class='hero-meta'>Click any tab for per-image details</div>")
    p.append("</div></div>")

    p.append("<div class='summary-grid'>")
    for i, r in enumerate(all_results):
        total = r["total_score"]
        results = r["results"]
        ok_count = sum(1 for m in results if m.global_score >= 60)
        bad_count = sum(1 for m in results if m.global_score < 40)
        sc_bg = _sc_bg(total)
        p.append(f"<a href='javascript:void(0)' onclick=\"document.getElementById('tab-img{i}').checked=true\" class='summary-card'>")
        p.append("<div class='summary-top'>")
        p.append(f"<div class='summary-name'>{r['name']}</div>")
        p.append(f"<div class='summary-score' style='color:{_sc(total)}'>{total:.1f}</div>")
        p.append("</div>")
        p.append("<div class='summary-bar-wrap'><div class='summary-bar'>")
        p.append(f"<div class='summary-bar-fill' style='width:{total}%;background:{sc_bg}'></div>")
        p.append("</div></div>")
        p.append("<div class='summary-stats'>")
        p.append(f"<div class='summary-stat'><div class='sv' style='color:var(--green)'>{ok_count}</div><div class='sl'>Pass</div></div>")
        p.append(f"<div class='summary-stat'><div class='sv' style='color:var(--red)'>{bad_count}</div><div class='sl'>Critical</div></div>")
        p.append("</div></a>")
    p.append("</div>")
    p.append("</div>")  # end summary panel

    # --- Per-Image Panels ---
    for i, r in enumerate(all_results):
        pid = f"panel-img{i}"
        p.append(f"<div id='{pid}' class='tab-panel'>")
        _render_image_section(p, r, f"tab-img{i}", is_first=(i == 0))
        p.append("</div>")

    # --- Comparison Panel ---
    if comparison_pairs:
        p.append("<div id='panel-compare' class='tab-panel'>")
        p.append("<div class='hero'><div class='hero-panel'>")
        p.append("<div class='hero-eyebrow'>COMPARISON</div>")
        p.append(f"<h1>{len(comparison_pairs)} Pair(s)</h1>")
        p.append("<div class='hero-meta'>Reference: first image in set</div>")
        p.append("</div></div>")

        for cp in comparison_pairs:
            p.append(f"<h2 class='section-hd'>{cp['test_name']} vs {cp['ref_name']}</h2>")
            delta = cp["total_score"] - cp["ref_total_score"]
            dc = "var(--green)" if delta >= 0 else "var(--red)"
            p.append(f"<div style='margin-bottom:12px;font-size:1rem;'>")
            p.append(f"<span style='color:var(--orange)'>Ref: {cp['ref_total_score']:.1f}</span> &rarr; ")
            p.append(f"<span style='color:var(--blue)'>Test: {cp['total_score']:.1f}</span> ")
            p.append(f"<span style='color:{dc};font-weight:800'>({delta:+.1f})</span>")
            p.append("</div>")

            p.append("<div class='panel'><div class='panel-inner'>")
            p.append("<table class='compare-table'>")
            p.append("<tr><th>Metric</th><th>Test</th><th>Ref</th><th>Delta</th><th>Diagnosis</th></tr>")
            ref_map = {m.name: m for m in cp["ref_results"]}
            for m in cp["results"]:
                ref = ref_map.get(m.name)
                rs = ref.global_score if ref else 0
                d = m.global_score - rs
                dc2 = "delta-up" if d >= 0 else "delta-down"
                cn = _METRIC_CN.get(m.name, m.name)
                p.append("<tr>")
                p.append(f"<td style='font-weight:800'>{cn}</td>")
                p.append(f"<td style='color:{_sc(m.global_score)}'><strong>{m.global_score:.1f}</strong></td>")
                p.append(f"<td>{rs:.1f}</td>")
                p.append(f"<td class='{dc2}'>{d:+.1f}</td>")
                p.append(f"<td style='font-size:0.8rem;color:var(--muted)'>{m.diagnosis}</td>")
                p.append("</tr>")
            p.append("</table></div></div>")
        p.append("</div>")  # end compare panel

    p.append("</div>")  # end tab-content

    # ── Tab switching JS ──
    p.append("<script>")
    p.append("(function(){")
    p.append("var radios=document.querySelectorAll('.tab-radio');")
    p.append("var panels=document.querySelectorAll('.tab-panel');")
    p.append("function switchTab(id){")
    p.append("panels.forEach(function(p){p.style.display='none';});")
    p.append("var panel=document.getElementById('panel-'+id.replace('tab-',''));")
    p.append("if(panel)panel.style.display='block';")
    p.append("}")
    p.append("radios.forEach(function(r){")
    p.append("r.addEventListener('change',function(){if(this.checked)switchTab(this.id);});")
    p.append("});")
    p.append("switchTab('tab-summary');")  # initial state
    p.append("})();")
    p.append("</script>")

    # ── Algorithm Reference ──
    p.append("<h2 class='section-hd'>Algorithm Reference</h2>")
    p.append("<div class='algo-grid'>")
    # Use first image's results for algo ref order
    ref_res = all_results[0]["results"] if all_results else []
    for m in ref_res:
        cn = _METRIC_CN.get(m.name, m.name)
        desc = _ALGO_DESCRIPTIONS.get(m.name, "")
        p.append(f"<div class='algo-item'><span class='aname'>{cn}</span> &mdash; {desc}</div>")
    p.append("</div>")

    p.append("<div class='footer'>Image Quality Scoring System &middot; AItuning</div>")
    p.append("</div></body></html>")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(p))


def _render_image_section(p: list, r: dict, tab_id: str, is_first: bool = False):
    """Render one image's full evaluation section."""
    name = r["name"]
    total = r["total_score"]
    results = r["results"]
    image_y = r.get("image_y")
    image_rgb = r.get("image_rgb")

    # Hero
    p.append("<div class='hero'><div class='hero-panel'>")
    p.append("<div class='hero-eyebrow'>PER-IMAGE EVALUATION</div>")
    p.append(f"<h1>{name}</h1>")
    sc_bg = _sc_bg(total)
    p.append(f"<div class='hero-score'><span class='val' style='color:{_sc(total)}'>{total:.1f}</span>")
    p.append("<span class='unit'>/ 100</span></div>")
    p.append("</div></div>")

    # Row: source + radar
    src_b64 = _make_source_figure(image_rgb, image_y)
    radar_b64 = _make_radar_figure(results, total)
    p.append("<div class='row2'>")
    p.append(f"<div class='panel'><div class='panel-inner'><img src='data:image/png;base64,{src_b64}'></div></div>")
    p.append(f"<div class='panel'><div class='panel-inner'><img src='data:image/png;base64,{radar_b64}'></div></div>")
    p.append("</div>")

    # Metrics grid
    p.append("<h2 class='section-hd'>Per-Metric Details</h2>")
    p.append("<div class='metrics-grid'>")
    for m in results:
        sim_b64 = _make_simulation_figure(image_y, m.heatmap, m.global_score) if image_y is not None else ""
        cn = _METRIC_CN.get(m.name, m.name)
        bc = _sc(m.global_score)
        p.append("<div class='metric-card'>")
        p.append("<div class='metric-head'>")
        p.append(f"<span class='mname'>{cn}</span>")
        p.append(f"<span class='mscore' style='color:{bc}'>{m.global_score:.1f}</span>")
        p.append("</div>")
        p.append("<div class='metric-bar-wrap'><div class='metric-bar'>")
        p.append(f"<div class='metric-bar-fill' style='width:{m.global_score}%;background:{bc}'></div>")
        p.append("</div></div>")
        if sim_b64:
            p.append(f"<div class='metric-img'><img src='data:image/png;base64,{sim_b64}'></div>")
        p.append(f"<div class='metric-diag'>{m.diagnosis}</div>")
        p.append("</div>")
    p.append("</div>")

    # Issues
    issues = [m for m in results if m.global_score < 60]
    p.append("<h2 class='section-hd'>Issues &amp; Diagnosis</h2>")
    if issues:
        p.append("<div class='diag-grid'>")
        for m in issues:
            cn = _METRIC_CN.get(m.name, m.name)
            p.append("<div class='diag-card'>")
            p.append(f"<div class='dname'>{cn} — {m.global_score:.1f}</div>")
            p.append(f"<div class='dtext'>{m.diagnosis}</div>")
            p.append("</div>")
        p.append("</div>")
    else:
        p.append("<div class='panel'><div class='panel-inner' style='color:var(--green);font-weight:700'>All metrics within normal range.</div></div>")


# Keep backward-compatible single-image entry point
def render_html(results, total_score, image_y, output_path,
                ref_results=None, ref_total_score=None, image_names=("Test", "Reference")):
    """Single-image HTML report (backward compat)."""
    all_results = [{"name": image_names[0], "results": results, "total_score": total_score,
                    "image_y": image_y, "image_rgb": None}]
    cmp = []
    if ref_results is not None:
        cmp = [{"test_name": image_names[0], "ref_name": image_names[1],
                "results": results, "total_score": total_score,
                "ref_results": ref_results, "ref_total_score": ref_total_score,
                "image_y": image_y, "image_rgb": None}]
    render_multi_report(all_results, cmp, output_path)
