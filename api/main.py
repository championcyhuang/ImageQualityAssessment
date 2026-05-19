"""FastAPI backend for Image Quality Assessment web admin.

Runs on localhost:8000. Serves:
- POST /upload   → save multipart files to 需要评估的图片/
- POST /evaluate → run scorer pipeline, generate HTML report
- GET  /reports/{id} → serve generated HTML report
"""

import asyncio
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Ensure project root is on path so `scorer` imports work
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scorer.pipeline import run_pipeline
from scorer.report.html_render import render_multi_report
from scorer.aggregate import DEFAULT_WEIGHTS

app = FastAPI(title="Image Quality Assessment API")

# CORS: allow Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories (relative to project root)
INPUT_DIR = _PROJECT_ROOT / "需要评估的图片"
REPORT_DIR = _PROJECT_ROOT / "评估报告"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".nv12"}

# In-memory task storage (dev only — resets on server restart)
eval_tasks: Dict[str, dict] = {}


class ImageSpec(BaseModel):
    path: str
    width: Optional[int] = None
    height: Optional[int] = None


class EvaluateRequest(BaseModel):
    images: List[ImageSpec]


# ------------------------------------------------------------------
# Exception handlers
# ------------------------------------------------------------------
@app.exception_handler(ValueError)
async def _value_error_handler(_req, exc: ValueError):
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.exception_handler(Exception)
async def _generic_exception_handler(_req, exc: Exception):
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    """Save uploaded files to 需要评估的图片/ and return paths."""
    saved = []
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {ext}. Allowed: {', '.join(SUPPORTED_EXTENSIONS)}",
            )
        dest = INPUT_DIR / (f.filename or "unnamed")
        content = await f.read()
        with open(dest, "wb") as out:
            out.write(content)
        saved.append(str(dest.relative_to(_PROJECT_ROOT)))
    return {"saved": saved}


def _do_evaluate(task_id: str, images: List[ImageSpec]):
    """Background task: run scorer for each image and update progress."""
    report_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out_dir = REPORT_DIR / report_id
    out_dir.mkdir(parents=True, exist_ok=True)

    task = eval_tasks[task_id]
    task["report_id"] = report_id
    task["report_url"] = f"/reports/{report_id}"

    all_results = []
    failed = []

    for i, spec in enumerate(images):
        img_path = _PROJECT_ROOT / spec.path
        if not img_path.exists():
            failed.append({"path": spec.path, "error": "File not found"})
            task["failed"] = failed.copy()
            task["completed"] = i + 1
            continue

        try:
            result = run_pipeline(
                str(img_path),
                width=spec.width,
                height=spec.height,
            )
        except Exception as exc:
            failed.append({"path": spec.path, "error": str(exc)})
            task["failed"] = failed.copy()
            task["completed"] = i + 1
            continue

        all_results.append({
            "name": img_path.name,
            "results": result["results"],
            "total_score": result["total_score"],
            "image_y": result["image_y"],
            "image_rgb": result.get("image_rgb"),
        })

        # Build per-image result for progress updates
        metrics = []
        for m in result["results"]:
            metrics.append({
                "name": m.name,
                "score": float(round(m.global_score, 1)),
                "weight": float(DEFAULT_WEIGHTS.get(m.name, 1.0)),
                "status": "excellent" if m.global_score >= 85 else "good" if m.global_score >= 60 else "warning",
            })
        task["results"].append({
            "name": img_path.name,
            "total_score": float(round(result["total_score"], 1)),
            "metrics": metrics,
        })
        task["completed"] = i + 1

    if not all_results:
        task["status"] = "error"
        return

    # Generate HTML report
    report_path = out_dir / "quality_report.html"
    render_multi_report(all_results, [], str(report_path))

    task["status"] = "partial" if failed else "done"


@app.post("/evaluate")
async def evaluate(req: EvaluateRequest, background_tasks: BackgroundTasks):
    """Queue evaluation task and return task_id for polling."""
    task_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    eval_tasks[task_id] = {
        "status": "running",
        "total": len(req.images),
        "completed": 0,
        "results": [],
        "failed": [],
        "report_id": None,
        "report_url": None,
    }

    # Run in thread pool so event loop isn't blocked
    background_tasks.add_task(_do_evaluate, task_id, req.images)

    return {"task_id": task_id, "status": "queued", "total": len(req.images)}


@app.get("/evaluate/status/{task_id}")
async def evaluate_status(task_id: str):
    """Poll evaluation progress."""
    task = eval_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.get("/reports/{report_id}", response_class=HTMLResponse)
async def get_report(report_id: str):
    """Serve generated HTML report."""
    report_file = REPORT_DIR / report_id / "quality_report.html"
    if not report_file.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    with open(report_file, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
