#!/usr/bin/env python3
"""Open-source text-to-video inference for a free Kaggle P100 (~16 GB)."""

from __future__ import annotations

import os
import gc
from pathlib import Path
from typing import Optional

import numpy as np
import torch

MODEL_CANDIDATES = (
    "ali-vilab/text-to-video-ms-1.7b",
    "damo-vilab/text-to-video-ms-1.7b",
)

OUTPUT_DIR = Path(os.environ.get("T2V_OUTPUT_DIR", "/kaggle/working/outputs"))
MAX_FRAMES = 24
MAX_SIDE = 320
MAX_STEPS = 40
MAX_PROMPT = 500

_pipe = None
_model_id = None
_device_note = ""


def gpu_report() -> dict:
    available = torch.cuda.is_available()
    info = {
        "cuda_available": available,
        "device": "cuda" if available else "cpu",
        "gpu_name": None,
        "vram_gb": None,
    }
    if available:
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(props.total_memory / (1024**3), 2)
    return info


def _export_mp4(frames, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = []
    raw = frames[0] if isinstance(frames, list) and frames and isinstance(frames[0], list) else frames
    for frame in raw:
        if hasattr(frame, "convert"):
            arr.append(np.array(frame.convert("RGB")))
        else:
            arr.append(np.asarray(frame))
    try:
        import imageio.v2 as imageio

        imageio.mimsave(str(path), arr, fps=fps, codec="libx264", quality=8)
        return
    except Exception:
        pass
    import cv2

    h, w = arr[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    for frame in arr:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def load_pipeline():
    global _pipe, _model_id, _device_note
    if _pipe is not None:
        return _pipe

    report = gpu_report()
    print("GPU report:", report)
    if not report["cuda_available"]:
        raise RuntimeError(
            "CUDA is not available. In the Kaggle notebook: Settings -> Accelerator -> GPU T4/P100, then restart."
        )

    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

    last_error = None
    dtype = torch.float16
    for model_id in MODEL_CANDIDATES:
        try:
            print(f"Loading {model_id} ...")
            pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, variant="fp16")
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            if hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload()
                _device_note = "fp16 + enable_model_cpu_offload + vae slicing"
            else:
                pipe.to("cuda")
                _device_note = "fp16 on cuda"
            _pipe = pipe
            _model_id = model_id
            print(f"Loaded {model_id} ({_device_note})")
            return _pipe
        except Exception as exc:
            last_error = exc
            print(f"Failed to load {model_id}: {exc}")
    raise RuntimeError(f"Could not load a text-to-video pipeline: {last_error}")


def validate_request(payload: dict) -> dict:
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt is required")
    if len(prompt) > MAX_PROMPT:
        raise ValueError(f"prompt exceeds {MAX_PROMPT} characters")

    negative = str(payload.get("negative_prompt") or "").strip()
    num_frames = int(payload.get("num_frames") or 16)
    height = int(payload.get("height") or 256)
    width = int(payload.get("width") or 256)
    fps = int(payload.get("fps") or 8)
    guidance = float(payload.get("guidance_scale") or 9.0)
    seed = int(payload.get("seed") if payload.get("seed") is not None else 42)
    steps = int(payload.get("num_inference_steps") or 25)

    if num_frames < 8 or num_frames > MAX_FRAMES:
        raise ValueError(f"num_frames must be between 8 and {MAX_FRAMES}")
    if height not in (256, 320) or width not in (256, 320):
        raise ValueError("height and width must be 256 or 320")
    if max(height, width) > MAX_SIDE:
        raise ValueError(f"resolution side cannot exceed {MAX_SIDE}")
    if fps < 4 or fps > 16:
        raise ValueError("fps must be between 4 and 16")
    if guidance < 1 or guidance > 20:
        raise ValueError("guidance_scale must be between 1 and 20")
    if steps < 10 or steps > MAX_STEPS:
        raise ValueError(f"num_inference_steps must be between 10 and {MAX_STEPS}")

    return {
        "prompt": prompt,
        "negative_prompt": negative,
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "fps": fps,
        "guidance_scale": guidance,
        "seed": seed,
        "num_inference_steps": steps,
    }


def generate_video(payload: dict, out_path: Optional[Path] = None, progress_cb=None) -> Path:
    params = validate_request(payload)
    if progress_cb:
        progress_cb("loading", 15, "Loading open-source video model")
    pipe = load_pipeline()
    if progress_cb:
        progress_cb("generating", 30, "Denoising video frames on GPU")

    generator = torch.Generator(device="cuda").manual_seed(params["seed"])
    kwargs = {
        "prompt": params["prompt"],
        "num_frames": params["num_frames"],
        "height": params["height"],
        "width": params["width"],
        "guidance_scale": params["guidance_scale"],
        "num_inference_steps": params["num_inference_steps"],
        "generator": generator,
    }
    if params["negative_prompt"]:
        kwargs["negative_prompt"] = params["negative_prompt"]

    result = pipe(**kwargs)
    frames = result.frames
    out = out_path or (OUTPUT_DIR / "clip.mp4")
    _export_mp4(frames, out, params["fps"])
    del result
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if progress_cb:
        progress_cb("completed", 100, "MP4 ready")
    return out


def model_info() -> dict:
    report = gpu_report()
    report["model_id"] = _model_id
    report["optimizations"] = _device_note
    report["loaded"] = _pipe is not None
    return report
