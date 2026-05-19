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
}

interface ImageQueueProps {
  items: QueueItem[];
  onRemove: (id: string) => void;
}

export default function ImageQueue({ items, onRemove }: ImageQueueProps) {
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

  return (
    <div className="space-y-2" role="list" aria-label="图片队列">
      {items.map((item) => (
        <div
          key={item.id}
          role="listitem"
          className="flex items-center justify-between rounded-[var(--radius-md)] px-3 py-2.5 border"
          style={{
            background: "var(--bg-secondary)",
            borderColor: "var(--border)",
          }}
        >
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
