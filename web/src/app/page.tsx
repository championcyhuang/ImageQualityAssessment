"use client";

import React, { useCallback, useRef, useState } from "react";
import UploadArea from "@/components/UploadArea";
import ImageQueue, { QueueItem } from "@/components/ImageQueue";
import NV12Modal from "@/components/NV12Modal";
import ReportViewer from "@/components/ReportViewer";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8002";

interface ReportResult {
  name: string;
  total_score: number;
  metrics: {
    name: string;
    score: number;
    weight: number;
    status: "excellent" | "good" | "warning";
  }[];
}

export default function Home() {
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [evaluating, setEvaluating] = useState(false);
  const [reportId, setReportId] = useState<string | null>(null);
  const [results, setResults] = useState<ReportResult[]>([]);
  const [nv12File, setNv12File] = useState<{ name: string; resolve: (dims: { w: number; h: number } | null) => void } | null>(null);
  const inputIdRef = useRef(0);

  const uploadFiles = useCallback(async (files: File[]) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json();
      alert(`上传失败: ${err.detail || err.error}`);
      return [] as string[];
    }
    const data = await res.json();
    return data.saved as string[];
  }, []);

  const processFiles = useCallback(async (files: File[]) => {
    const regular = files.filter((f) => !/\.nv12$/i.test(f.name));
    const nv12s = files.filter((f) => /\.nv12$/i.test(f.name));

    const newItems: QueueItem[] = [];

    if (regular.length) {
      const saved = await uploadFiles(regular);
      saved.forEach((path) => {
        newItems.push({
          id: `item-${++inputIdRef.current}`,
          name: path.split("/").pop() || path,
          path,
          status: "ready",
        });
      });
    }

    for (const file of nv12s) {
      const saved = await uploadFiles([file]);
      if (!saved.length) continue;
      const path = saved[0];

      const dims = await new Promise<{ w: number; h: number } | null>((resolve) => {
        setNv12File({
          name: file.name,
          resolve: (dims) => {
            resolve(dims);
            setNv12File(null);
          },
        });
      });

      if (dims && dims.w > 0 && dims.h > 0) {
        newItems.push({
          id: `item-${++inputIdRef.current}`,
          name: file.name,
          path,
          width: dims.w,
          height: dims.h,
          status: "ready",
        });
      }
    }

    if (newItems.length) {
      setQueue((prev) => [...prev, ...newItems]);
    }
  }, [uploadFiles]);

  const handleEvaluate = useCallback(async () => {
    const readyItems = queue.filter((q) => q.status === "ready");
    if (!readyItems.length) return;

    setEvaluating(true);
    setQueue((prev) =>
      prev.map((q) => (q.status === "ready" ? { ...q, status: "evaluating" as const } : q))
    );

    try {
      const res = await fetch(`${API_BASE}/evaluate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          images: readyItems.map((q) => ({
            path: q.path,
            width: q.width,
            height: q.height,
          })),
        }),
      });

      const data = await res.json();

      if (data.status === "error") {
        setQueue((prev) =>
          prev.map((q) =>
            q.status === "evaluating" ? { ...q, status: "error" as const, error: "评估失败" } : q
          )
        );
        alert("评估失败: " + (data.error || "未知错误"));
        return;
      }

      setReportId(data.id);
      setResults(data.results || []);

      setQueue((prev) =>
        prev.map((q) => {
          if (q.status !== "evaluating") return q;
          const found = data.failed?.find((f: any) => f.path === q.path);
          if (found) {
            return { ...q, status: "error" as const, error: found.error };
          }
          return { ...q, status: "done" as const };
        })
      );
    } catch (err: any) {
      setQueue((prev) =>
        prev.map((q) =>
          q.status === "evaluating" ? { ...q, status: "error" as const, error: err.message } : q
        )
      );
      alert("评估请求失败: " + err.message);
    } finally {
      setEvaluating(false);
    }
  }, [queue]);

  const handleRemove = useCallback((id: string) => {
    setQueue((prev) => prev.filter((q) => q.id !== id));
  }, []);

  const hasReady = queue.some((q) => q.status === "ready");

  return (
    <div className="flex flex-col h-screen" style={{ background: "var(--bg-primary)" }}>
      {/* Topbar */}
      <header
        className="shrink-0 border-b px-6 py-3 flex items-center justify-between"
        style={{ borderColor: "var(--border)", background: "rgba(251,247,241,0.86)" }}
      >
        <div className="flex items-center gap-2.5">
          <div
            className="w-8 h-8 rounded-lg grid place-items-center text-white font-black text-sm"
            style={{ background: "linear-gradient(140deg, var(--score-excellent) 0%, var(--accent) 100%)" }}
          >
            IQ
          </div>
          <div>
            <h1 className="text-sm font-black" style={{ color: "var(--text-primary)" }}>
              图像质量评估
            </h1>
            <p className="text-xs" style={{ color: "var(--text-secondary)" }}>
              13 项指标 · 综合评分
            </p>
          </div>
        </div>
        <button
          onClick={handleEvaluate}
          disabled={!hasReady || evaluating}
          className="px-5 py-2 rounded-[var(--radius-md)] text-sm font-bold text-white transition-all disabled:opacity-40 disabled:cursor-not-allowed hover:brightness-105"
          style={{ background: "var(--accent)" }}
        >
          {evaluating ? "评估中..." : "开始评估"}
        </button>
      </header>

      {/* Main content */}
      <main className="flex-1 overflow-hidden flex flex-col lg:flex-row">
        {/* Left: upload + queue */}
        <aside
          className="w-full lg:w-80 shrink-0 border-r overflow-y-auto p-4 space-y-4"
          style={{ borderColor: "var(--border)" }}
        >
          <UploadArea onFilesSelect={processFiles} disabled={evaluating} />
          <div>
            <p className="text-xs font-bold uppercase tracking-wide mb-2" style={{ color: "var(--text-secondary)" }}>
              待评估队列
            </p>
            <ImageQueue items={queue} onRemove={handleRemove} />
          </div>
        </aside>

        {/* Right: report */}
        <section className="flex-1 overflow-y-auto p-4 lg:p-6" aria-live="polite">
          <ReportViewer results={results} reportId={reportId} loading={evaluating} />
        </section>
      </main>

      {/* NV12 Modal */}
      {nv12File && (
        <NV12Modal
          fileName={nv12File.name}
          onConfirm={(w, h) => nv12File.resolve({ w, h })}
          onCancel={() => {
            nv12File.resolve(null);
            setNv12File(null);
          }}
        />
      )}
    </div>
  );
}
