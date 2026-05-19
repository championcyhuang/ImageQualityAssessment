"use client";

import React from "react";

const API_BASE = "/api";

interface Metric {
  name: string;
  score: number;
  weight: number;
  status: "excellent" | "good" | "warning";
}

interface ReportResult {
  name: string;
  total_score: number;
  metrics: Metric[];
}

interface ReportViewerProps {
  results: ReportResult[];
  reportId: string | null;
  loading: boolean;
}

export default function ReportViewer({ results, reportId, loading }: ReportViewerProps) {
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-20" role="status" aria-label="评估中">
        <div className="w-8 h-8 border-2 border-t-transparent rounded-full animate-spin mb-3"
          style={{ borderColor: "var(--accent)", borderTopColor: "transparent" }}
        />
        <p className="text-sm font-medium" style={{ color: "var(--text-secondary)" }}>
          评估中...
        </p>
      </div>
    );
  }

  if (!results.length) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <p className="text-sm mb-2" style={{ color: "var(--text-secondary)" }}>
          暂无报告，先上传并评估图片
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {results.map((r) => (
        <div key={r.name}>
          {/* Total score */}
          <div className="mb-4">
            <p className="text-xs font-medium uppercase tracking-wide mb-1" style={{ color: "var(--text-secondary)" }}>
              综合评分 · {r.name}
            </p>
            <div className="flex items-baseline gap-2">
              <span className="text-5xl font-black" style={{ color: scoreColor(r.total_score), fontFamily: "var(--font-mono)" }}>
                {r.total_score.toFixed(1)}
              </span>
              <span className="text-sm font-medium" style={{ color: "var(--text-secondary)" }}>/ 100</span>
            </div>
          </div>

          {/* Metrics grouped by status */}
          <div className="space-y-3">
            <MetricGroup title="优秀 (≥85)" metrics={r.metrics.filter((m) => m.status === "excellent")} color="var(--score-excellent)" />
            <MetricGroup title="良好 (60-85)" metrics={r.metrics.filter((m) => m.status === "good")} color="var(--score-good)" />
            <MetricGroup title="需关注 (<60)" metrics={r.metrics.filter((m) => m.status === "warning")} color="var(--score-warning)" />
          </div>
        </div>
      ))}

      {/* Full report iframe */}
      {reportId && (
        <div className="mt-6">
          <p className="text-xs font-medium uppercase tracking-wide mb-2" style={{ color: "var(--text-secondary)" }}>
            完整报告
          </p>
          <div
            className="rounded-[var(--radius-lg)] overflow-hidden border"
            style={{ borderColor: "var(--border)" }}
          >
            <iframe
              src={`${API_BASE}/reports/${reportId}`}
              className="w-full h-[600px]"
              title="评估报告"
            />
          </div>
          <a
            href={`${API_BASE}/reports/${reportId}`}
            download={`report_${reportId}.html`}
            className="inline-block mt-2 text-sm font-medium hover:underline"
            style={{ color: "var(--accent)" }}
          >
            下载报告
          </a>
        </div>
      )}
    </div>
  );
}

function MetricGroup({ title, metrics, color }: { title: string; metrics: Metric[]; color: string }) {
  if (!metrics.length) return null;
  return (
    <div>
      <p className="text-xs font-bold mb-2" style={{ color }}>
        {title}
      </p>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
        {metrics.map((m) => (
          <div
            key={m.name}
            className="rounded-[var(--radius-md)] border px-3 py-2"
            style={{ background: "var(--bg-secondary)", borderColor: "var(--border)" }}
          >
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-medium" style={{ color: "var(--text-primary)" }}>
                {metricCN(m.name)}
              </span>
              <span className="text-sm font-bold" style={{ color, fontFamily: "var(--font-mono)" }}>
                {m.score.toFixed(1)}
              </span>
            </div>
            <div className="h-1 rounded-full overflow-hidden" style={{ background: "rgba(22,49,58,0.06)" }}>
              <div
                className="h-full rounded-full"
                style={{ width: `${Math.min(m.score, 100)}%`, background: color }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function scoreColor(score: number): string {
  if (score >= 85) return "var(--score-excellent)";
  if (score >= 60) return "var(--score-good)";
  return "var(--score-warning)";
}

const METRIC_CN: Record<string, string> = {
  exposure: "曝光",
  brightness: "亮度",
  contrast: "对比度",
  color_accuracy: "色彩准确度",
  white_balance: "白平衡",
  sharpness: "清晰度",
  noise: "噪声",
  dynamic_range: "动态范围",
  texture_preservation: "纹理保留",
  uniformity: "均匀性",
  fringing: "色散",
  saturation: "饱和度",
  distortion: "畸变",
};

function metricCN(name: string): string {
  return METRIC_CN[name] || name;
}
