#!/usr/bin/env python3
"""Temporary FastAPI backend for a free Colab GPU runtime."""

from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from generator import (
    OUTPUT_DIR,
    GenerationError,
    friendly_error,
    generate_video,
    gpu_report,
    load_pipeline,
    model_info,
    validate_request,
)

HOST = os.environ.get("T2V_HOST", "0.0.0.0")
PORT = int(os.environ.get("T2V_PORT", "8000"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_ORIGINS = [
    "https://sgue19000.github.io",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]

app = FastAPI(title="Lumen Clip Colab API", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.github\.io|http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()
_busy = threading.Event()


class GenerateBody(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500)
    negative_prompt: Optional[str] = ""
    num_frames: int = 8
    height: int = 256
    width: int = 256
    fps: int = 8
    steps: Optional[int] = 15
    num_inference_steps: Optional[int] = None
    guidance_scale: float = 9.0
    seed: int = 42


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    out = {
        "job_id": job.get("job_id"),
        "prompt": job.get("prompt"),
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "message": job.get("message"),
        "created_at": job.get("created_at"),
        "completed_at": job.get("completed_at"),
        "error": job.get("error"),
    }
    if job.get("status") == "completed":
        out["video_url"] = f"/video/{job['job_id']}"
    return out


def _set_job(job_id: str, **fields: Any) -> None:
    with _lock:
        job = _jobs.setdefault(job_id, {"job_id": job_id})
        job.update(fields)


def _run_job(job_id: str, payload: dict[str, Any]) -> None:
    try:
        _set_job(job_id, status="loading", progress=10, message="Loading model")

        def cb(status: str, progress: int, message: str) -> None:
            _set_job(job_id, status=status, progress=progress, message=message)

        out = OUTPUT_DIR / f"{job_id}.mp4"
        generate_video(payload, out_path=out, progress_cb=cb)
        _set_job(job_id, status="completed", progress=100, message="Done", output_path=str(out), completed_at=time.time(), error=None)
    except Exception as exc:
        _set_job(job_id, status="failed", progress=0, message="failed", error=friendly_error(exc), completed_at=time.time())
    finally:
        _busy.clear()


@app.get("/health")
def health():
    info = model_info()
    report = gpu_report()
    return {
        "ok": True,
        "status": "ok",
        "gpu": report.get("gpu"),
        "cuda_available": report.get("cuda_available"),
        "vram_gb": report.get("vram_gb"),
        "torch": report.get("torch"),
        "cuda_version": report.get("cuda_version"),
        "model_id": info.get("model_id"),
        "loaded": info.get("loaded"),
        "temporary_session": True,
        "backend": "colab",
        "busy": _busy.is_set(),
    }


@app.post("/generate")
def generate(body: GenerateBody):
    payload = body.model_dump()
    if payload.get("num_inference_steps") and not payload.get("steps"):
        payload["steps"] = payload["num_inference_steps"]
    try:
        params = validate_request(payload)
    except GenerationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if _busy.is_set():
        raise HTTPException(status_code=409, detail="The Colab GPU is already generating a video. Wait for that job to finish.")
    job_id = uuid.uuid4().hex[:12]
    now = time.time()
    _set_job(job_id, prompt=params["prompt"], status="queued", progress=1, message="Queued", created_at=now, completed_at=None, output_path=None, error=None)
    _busy.set()
    threading.Thread(target=_run_job, args=(job_id, params), daemon=True).start()
    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}")
def status(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job. Start a new generation.")
    return _public_job(job)


@app.get("/video/{job_id}")
def video(job_id: str):
    path = OUTPUT_DIR / f"{job_id}.mp4"
    if not path.exists() or path.stat().st_size < 1024:
        raise HTTPException(status_code=404, detail="Video is not ready yet.")
    return FileResponse(path, media_type="video/mp4", filename=f"{job_id}.mp4", headers={"Content-Disposition": f'inline; filename="{job_id}.mp4"'})


def preload() -> None:
    if os.environ.get("T2V_SKIP_PRELOAD") == "1":
        print("Skipping model preload")
        return
    print("Preloading model")
    load_pipeline()
    print(model_info())


if __name__ == "__main__":
    import uvicorn
    preload()
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
