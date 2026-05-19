"use client";

import React from "react";

export interface QueueItem {
  id: string;
  name: string;
  path: string;
  width?: number;
  height?: number;
  status: "ready" | "evaluating" | "done" | "error";
  error?: string;
  previewUrl?: string;
}

interface ImageQueueProps {
  items: QueueItem[];
  onRemove: (id: string) => void;
  progress?: { completed: number; total: number };
}

export default function ImageQueue({ items, onRemove, progress }: ImageQueueProps) {
  if (items.length === 0) {
    return (
      <div
        className="rounded-[var(--radius-lg)] p-4 text-center text-sm"
        style={{ color: "var(--text-secondary)", background: "var(--bg-secondary)" }}
      >
        暂无图片，从左侧上传
      </div>
    );
  }

  const showProgress = progress && progress.total > 0 && progress.completed < progress.total;

  return (
    <div className="space-y-2" role="list" aria-label="图片队列">
      {showProgress && (
        <div className="mb-3">
          <div className="flex justify-between text-xs mb-1" style={{ color: "var(--text-secondary)" }}>
            <span>评估进度</span>
            <span>{progress!.completed} / {progress!.total}</span>
          </div>
          <div className="h-1.5 rounded-full overflow-hidden" style={{ background: "var(--border)" }}>
            <div
              className="h-full rounded-full transition-all duration-300"
              style={{
                width: `${(progress!.completed / progress!.total) * 100}%`,
                background: "var(--accent)",
              }}
            />
          </div>
        </div>
      )}
      {items.map((item) => (
        <div
          key={item.id}
          role="listitem"
          className="flex items-center gap-3 rounded-[var(--radius-md)] px-3 py-2.5 border"
          style={{
            background: "var(--bg-secondary)",
            borderColor: "var(--border)",
          }}
        >
          {item.previewUrl && (
            <img
              src={item.previewUrl}
              alt={item.name}
              className="w-12 h-12 rounded object-cover shrink-0"
            />
          )}
          <div className="min-w-0 flex-1">
            <p className="text-sm font-medium truncate" style={{ color: "var(--text-primary)" }}>
              {item.name}
            </p>
            <p className="text-xs truncate" style={{ color: "var(--text-secondary)" }}>
              {item.width && item.height ? `${item.width}x${item.height}` : "自动检测"}
              {" · "}
              <StatusBadge status={item.status} />
            </p>
            {item.error && (
              <p className="text-xs mt-0.5" style={{ color: "var(--score-warning)" }}>
                {item.error}
              </p>
            )}
          </div>
          <button
            onClick={() => onRemove(item.id)}
            className="ml-2 w-7 h-7 flex items-center justify-center rounded text-sm hover:bg-black/5 transition-colors shrink-0"
            style={{ color: "var(--text-secondary)" }}
            aria-label={`删除 ${item.name}`}
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );
}

function StatusBadge({ status }: { status: QueueItem["status"] }) {
  const map: Record<string, { label: string; color: string }> = {
    ready: { label: "就绪", color: "var(--text-secondary)" },
    evaluating: { label: "评估中", color: "var(--accent)" },
    done: { label: "完成", color: "var(--score-excellent)" },
    error: { label: "失败", color: "var(--score-warning)" },
  };
  const s = map[status] || map.ready;
  return (
    <span className="font-medium" style={{ color: s.color }}>
      {s.label}
    </span>
  );
}
