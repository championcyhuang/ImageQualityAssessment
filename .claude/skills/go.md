---
name: go
description: Use when the user types "go" or asks to run image quality evaluation — triggers automated batch evaluation of images using the imglab venv Python, with automatic comparison logic when multiple images are detected
---

# Go — 图像质量评估

直接运行:

```bash
"F:/AI/AItuning/.venv/Scripts/python.exe" F:/AI/AItuning/go_eval.py
```

## 输入/输出

| | 路径 |
|---|---|
| 输入 | `F:\AI\AItuning\需要评估的图片` (.png / .jpg / .jpeg / .nv12) |
| 输出 | `F:\AI\AItuning\评估报告\{timestamp}\quality_report.html` |
| 母版 | `F:\AI\AItuning\评估报告母版\report_master.html` |

## 行为

- **1 张图片** → 单张评估，Summary + 单图详情标签页
- **2+ 张图片** → 每张独立评估 + Comparison 对比标签页
- 标签页：Summary（总览） → 各图详情（Hero + 原图 + 雷达图 + 12指标仿真图 + 诊断） → Comparison（逐项 Delta）
- 报告采用 Champion 设计风格（暖白配色、原图 RGB 展示、仿真图叠加、卡片式布局）13 项指标
- 评估完成后**不删除**输入图片
