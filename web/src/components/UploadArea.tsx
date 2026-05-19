"use client";

import React, { useCallback } from "react";

interface UploadAreaProps {
  onFilesSelect: (files: File[]) => void;
  disabled?: boolean;
}

export default function UploadArea({ onFilesSelect, disabled }: UploadAreaProps) {
  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      if (disabled) return;
      const files = Array.from(e.dataTransfer.files).filter((f) =>
        /\.(png|jpe?g|nv12)$/i.test(f.name)
      );
      if (files.length) onFilesSelect(files);
    },
    [disabled, onFilesSelect]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const handleClick = useCallback(() => {
    if (disabled) return;
    const input = document.createElement("input");
    input.type = "file";
    input.multiple = true;
    input.accept = ".png,.jpg,.jpeg,.nv12";
    input.onchange = () => {
      const files = Array.from(input.files || []).filter((f) =>
        /\.(png|jpe?g|nv12)$/i.test(f.name)
      );
      if (files.length) onFilesSelect(files);
    };
    input.click();
  }, [disabled, onFilesSelect]);

  return (
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onClick={handleClick}
      className={`
        border-2 border-dashed rounded-[var(--radius-lg)] p-8 text-center cursor-pointer
        transition-colors duration-200
        ${disabled ? "opacity-50 cursor-not-allowed" : "hover:border-[var(--accent)] hover:bg-[var(--bg-secondary)]"}
      `}
      style={{ borderColor: "var(--border)" }}
      role="region"
      aria-label="图片上传区"
    >
      <div className="text-4xl mb-3" aria-hidden>+</div>
      <p className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
        拖图片到这里，或点击上传
      </p>
      <p className="text-xs mt-2" style={{ color: "var(--text-secondary)" }}>
        支持 PNG、JPG、NV12
      </p>
    </div>
  );
}
