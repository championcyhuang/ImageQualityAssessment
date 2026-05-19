"use client";

import React, { useState } from "react";

interface NV12ModalProps {
  fileName: string;
  onConfirm: (width: number, height: number) => void;
  onCancel: () => void;
}

export default function NV12Modal({ fileName, onConfirm, onCancel }: NV12ModalProps) {
  const [width, setWidth] = useState("");
  const [height, setHeight] = useState("");
  const [error, setError] = useState("");

  const handleConfirm = () => {
    const w = parseInt(width, 10);
    const h = parseInt(height, 10);
    if (!w || w <= 0 || !h || h <= 0) {
      setError("请输入有效的宽度和高度（正整数）");
      return;
    }
    onConfirm(w, h);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30">
      <div
        className="w-full max-w-sm rounded-[var(--radius-lg)] p-6 shadow-lg"
        style={{ background: "var(--bg-secondary)" }}
      >
        <h3 className="text-lg font-bold mb-1" style={{ color: "var(--text-primary)" }}>
          输入 NV12 尺寸
        </h3>
        <p className="text-sm mb-4" style={{ color: "var(--text-secondary)" }}>
          {fileName} 需要 width 和 height 才能解析
        </p>

        <div className="flex gap-3 mb-4">
          <div className="flex-1">
            <label className="block text-xs font-medium mb-1" style={{ color: "var(--text-secondary)" }}>
              宽度
            </label>
            <input
              type="number"
              value={width}
              onChange={(e) => setWidth(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleConfirm()}
              className="w-full rounded-[var(--radius-md)] border px-3 py-2 text-sm outline-none focus:border-[var(--accent)]"
              style={{ borderColor: "var(--border)", background: "var(--bg-primary)", color: "var(--text-primary)" }}
              placeholder="1920"
              autoFocus
            />
          </div>
          <div className="flex-1">
            <label className="block text-xs font-medium mb-1" style={{ color: "var(--text-secondary)" }}>
              高度
            </label>
            <input
              type="number"
              value={height}
              onChange={(e) => setHeight(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleConfirm()}
              className="w-full rounded-[var(--radius-md)] border px-3 py-2 text-sm outline-none focus:border-[var(--accent)]"
              style={{ borderColor: "var(--border)", background: "var(--bg-primary)", color: "var(--text-primary)" }}
              placeholder="1080"
            />
          </div>
        </div>

        {error && (
          <p className="text-xs mb-3" style={{ color: "var(--score-warning)" }}>{error}</p>
        )}

        <div className="flex gap-2 justify-end">
          <button
            onClick={onCancel}
            className="px-4 py-2 rounded-[var(--radius-md)] text-sm font-medium transition-colors"
            style={{ background: "var(--bg-primary)", color: "var(--text-secondary)" }}
          >
            取消
          </button>
          <button
            onClick={handleConfirm}
            className="px-4 py-2 rounded-[var(--radius-md)] text-sm font-medium text-white transition-colors"
            style={{ background: "var(--accent)" }}
          >
            确认
          </button>
        </div>
      </div>
    </div>
  );
}
