#!/usr/bin/env python3
"""Temporary REST API for the Kaggle GPU notebook. Not a permanent host."""

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

from generator import OUTPUT_DIR, generate_video, load_pipeline, model_info, validate_request

HOST = os.environ.get("T2V_HOST", "0.0.0.0")
PORT = int(os.environ.get("T2V_PORT", "8000"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Lumen Clip Kaggle API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()
_worker_busy = threading.Event()


class GenerateBody(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500)
    negative_prompt: Optional[str] = ""
    num_frames: int = 16
    height: int = 256
    width: int = 256
    fps: int = 8
    guidance_scale: float = 9.0
    seed: int = 42
    num_inference_steps: int = 25


def _set_job(job_id: str, **fields: Any) -> None:
    with _lock:
        job = _jobs.setdefault(job_id, {"job_id": job_id})
        job.update(fields)
        job["updated_at"] = time.time()


def _run_job(job_id: str, payload: dict) -> None:
    try:
        _set_job(job_id, status="loading", progress=10, message="Warming model")

        def cb(status: str, progress: int, message: str) -> None:
            _set_job(job_id, status=status, progress=progress, message=message)

        out = OUTPUT_DIR / f"{job_id}.mp4"
        generate_video(payload, out_path=out, progress_cb=cb)
        _set_job(job_id, status="completed", progress=100, message="Done", video_url=f"/video/{job_id}")
    except Exception as exc:
        _set_job(job_id, status="failed", progress=0, error=str(exc), message="failed")
    finally:
        _worker_busy.clear()


@app.get("/health")
def health():
    info = model_info()
    return {"ok": True, "status": "ok", "temporary_session": True, **info}


@app.post("/generate")
def generate(body: GenerateBody):
    payload = body.model_dump()
    try:
        validate_request(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if _worker_busy.is_set():
        raise HTTPException(
            status_code=409,
            detail="A generation job is already running on this Kaggle GPU. Wait for it to finish.",
        )

    job_id = uuid.uuid4().hex[:12]
    _set_job(job_id, status="queued", progress=1, message="Queued")
    _worker_busy.set()
    threading.Thread(target=_run_job, args=(job_id, payload), daemon=True).start()
    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}")
def status(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return job


@app.get("/video/{job_id}")
def video(job_id: str):
    path = OUTPUT_DIR / f"{job_id}.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video not ready")
    return FileResponse(path, media_type="video/mp4", filename=f"{job_id}.mp4")


def preload() -> None:
    print("Preloading model so the first web request is faster...")
    load_pipeline()
    print(model_info())


if __name__ == "__main__":
    import uvicorn

    preload()
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
