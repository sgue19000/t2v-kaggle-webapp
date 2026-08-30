#!/usr/bin/env python3
"""Temporary Colab text-to-video generator. TextToVideoSDPipeline."""
from __future__ import annotations

import gc
import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Optional

MODEL_CANDIDATES = (
    "ali-vilab/text-to-video-ms-1.7b",
    "damo-vilab/text-to-video-ms-1.7b",
)
MAX_PROMPT = 500
ALLOWED_FRAMES = (8, 16)
ALLOWED_SIDES = (256,)
ALLOWED_FPS = (6, 8, 12)
MIN_STEPS = 10
MAX_STEPS = 25
DEFAULTS = {
    "num_frames": 8,
    "width": 256,
    "height": 256,
    "fps": 8,
    "steps": 15,
    "guidance_scale": 9.0,
    "seed": 42,
}
_pipe = None
_model_id: Optional[str] = None
_device_note = ""
_load_lock = threading.Lock()
_DIFFUSERS_READY = False


def _patch_py313_metadata() -> None:
    """Python 3.13 packages_distributions() can hang for minutes on Colab."""
    dummy = lambda: {}  # noqa: E731
    try:
        import importlib.metadata as md
        md.packages_distributions = dummy  # type: ignore[method-assign]
    except Exception:
        pass
    try:
        import importlib_metadata as md2
        md2.packages_distributions = dummy  # type: ignore[method-assign]
    except Exception:
        pass


def _import_diffusers():
    global _DIFFUSERS_READY
    _patch_py313_metadata()
    if _DIFFUSERS_READY:
        import diffusers
        return diffusers
    print("Importing Diffusers...")
    sys.stdout.flush()
    import diffusers
    _DIFFUSERS_READY = True
    print("Diffusers imported")
    sys.stdout.flush()
    return diffusers


def _default_output_dir() -> Path:
    env = os.environ.get("T2V_OUTPUT_DIR")
    if env:
        return Path(env)
    if Path("/content").exists():
        return Path("/content/outputs")
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/outputs")
    return Path("./outputs")


OUTPUT_DIR = _default_output_dir()


class GenerationError(Exception):
    """Safe user-facing generation failure."""


def gpu_report() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {"cuda_available": False, "gpu": None, "vram_gb": None, "device": "cpu", "torch": None, "cuda_version": None}
    available = bool(torch.cuda.is_available())
    info: dict[str, Any] = {
        "cuda_available": available,
        "gpu": None,
        "vram_gb": None,
        "device": "cuda" if available else "cpu",
        "torch": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
    }
    if available:
        info["gpu"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    return info


def _as_numpy(frames: Any):
    import numpy as np
    if frames is None:
        return None
    if hasattr(frames, "detach"):
        frames = frames.detach().cpu().numpy()
    if isinstance(frames, np.ndarray):
        return frames
    return None


def _to_uint8_hwc(frame: Any):
    import numpy as np
    if hasattr(frame, "convert"):
        return np.asarray(frame.convert("RGB"), dtype=np.uint8)
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        amax = float(np.nanmax(arr)) if arr.size else 0.0
        if amax <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _to_pil(frame: Any):
    from PIL import Image
    if hasattr(frame, "convert"):
        return frame.convert("RGB")
    return Image.fromarray(_to_uint8_hwc(frame), mode="RGB")


def _frame_count(obj: Any) -> int:
    import numpy as np
    if obj is None:
        return 0
    if isinstance(obj, np.ndarray):
        return int(obj.shape[0]) if obj.ndim >= 1 else 0
    if hasattr(obj, "__len__"):
        try:
            return len(obj)
        except TypeError:
            return 0
    return 0


def _normalize_frames(frames: Any) -> list:
    import numpy as np
    if frames is None:
        raise GenerationError("The model returned no frames.")
    if hasattr(frames, "frames"):
        frames = frames.frames
    arr = _as_numpy(frames)
    if arr is not None:
        if arr.ndim == 5:
            arr = arr[0]
        if arr.ndim != 4:
            raise GenerationError("Unexpected video tensor shape from the model.")
        if arr.shape[1] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (0, 2, 3, 1))
        pil_frames = [_to_pil(arr[i]) for i in range(arr.shape[0])]
        if _frame_count(pil_frames) == 0:
            raise GenerationError("The model returned an empty frame list.")
        return pil_frames
    if isinstance(frames, (list, tuple)):
        if _frame_count(frames) == 0:
            raise GenerationError("The model returned an empty frame list.")
        first = frames[0]
        first_arr = _as_numpy(first)
        if isinstance(first, (list, tuple)) or (first_arr is not None and getattr(first_arr, "ndim", 0) == 4):
            frames = first
        if _frame_count(frames) == 0:
            raise GenerationError("The model returned an empty frame list.")
        return [_to_pil(frame) for frame in frames]
    raise GenerationError("Could not read frames from the model output.")


def _export_mp4(frames: Any, path: Path, fps: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    pil_frames = _normalize_frames(frames)
    try:
        from diffusers.utils import export_to_video
        export_to_video(pil_frames, str(path), fps=int(fps))
    except Exception:
        try:
            import imageio.v2 as imageio
            imageio.mimsave(str(path), [_to_uint8_hwc(f) for f in pil_frames], fps=int(fps), codec="libx264")
        except Exception as exc:
            raise GenerationError("Could not write an MP4 from the generated frames.") from exc
    if not path.exists() or path.stat().st_size < 256:
        raise GenerationError("MP4 export produced an empty or missing file.")
    return path


def validate_request(payload: dict[str, Any]) -> dict[str, Any]:
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise GenerationError("Prompt is empty. Type a short English scene description.")
    if len(prompt) > MAX_PROMPT:
        raise GenerationError(f"Prompt is too long. Use {MAX_PROMPT} characters or fewer.")
    negative = str(payload.get("negative_prompt") or "").strip()
    num_frames = int(payload.get("num_frames") or DEFAULTS["num_frames"])
    height = int(payload.get("height") or DEFAULTS["height"])
    width = int(payload.get("width") or DEFAULTS["width"])
    fps = int(payload.get("fps") or DEFAULTS["fps"])
    if payload.get("steps") is not None:
        steps = int(payload.get("steps"))
    else:
        steps = int(payload.get("num_inference_steps") or DEFAULTS["steps"])
    guidance = float(payload.get("guidance_scale") if payload.get("guidance_scale") is not None else DEFAULTS["guidance_scale"])
    seed = int(payload.get("seed") if payload.get("seed") is not None else DEFAULTS["seed"])
    if num_frames not in ALLOWED_FRAMES:
        raise GenerationError("Frame count must be 8 or 16 on a free Colab GPU.")
    if height not in ALLOWED_SIDES or width not in ALLOWED_SIDES:
        raise GenerationError("Resolution must be 256x256 on a free Colab GPU.")
    if fps not in ALLOWED_FPS:
        raise GenerationError("FPS must be 6, 8, or 12.")
    if steps < MIN_STEPS or steps > MAX_STEPS:
        raise GenerationError(f"Steps must be between {MIN_STEPS} and {MAX_STEPS}.")
    if guidance < 1 or guidance > 15:
        raise GenerationError("Guidance scale must be between 1 and 15.")
    return {
        "prompt": prompt,
        "negative_prompt": negative,
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "fps": fps,
        "steps": steps,
        "guidance_scale": guidance,
        "seed": seed,
    }


def load_pipeline():
    global _pipe, _model_id, _device_note
    if _pipe is not None:
        print("Model loaded")
        return _pipe
    with _load_lock:
        if _pipe is not None:
            print("Model loaded")
            return _pipe
        import torch
        report = gpu_report()
        print("GPU detected:", report.get("gpu") or "none")
        sys.stdout.flush()
        if not report["cuda_available"]:
            raise GenerationError("GPU unavailable. In Colab open Runtime -> Change runtime type -> GPU, then Run all.")
        _import_diffusers()
        from diffusers import DPMSolverMultistepScheduler, TextToVideoSDPipeline
        last_error: Optional[Exception] = None
        for model_id in MODEL_CANDIDATES:
            try:
                print("Loading TextToVideoSDPipeline...")
                print(f"Loading {model_id}")
                sys.stdout.flush()
                pipe = TextToVideoSDPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
                if hasattr(pipe, "enable_vae_slicing"):
                    pipe.enable_vae_slicing()
                if hasattr(pipe, "enable_vae_tiling"):
                    try:
                        pipe.enable_vae_tiling()
                    except Exception:
                        pass
                if hasattr(pipe, "enable_model_cpu_offload"):
                    pipe.enable_model_cpu_offload()
                    _device_note = "fp16 + cpu offload + vae slicing"
                else:
                    pipe.to("cuda")
                    _device_note = "fp16 on cuda"
                _pipe = pipe
                _model_id = model_id
                print("Model loaded")
                print(f"Loaded {model_id} ({_device_note})")
                sys.stdout.flush()
                return _pipe
            except GenerationError:
                raise
            except Exception as exc:
                last_error = exc
                print(f"Failed to load {model_id}: {exc}")
        raise GenerationError("Could not load the open-source text-to-video model from Hugging Face.") from last_error


def _clear_cuda() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def generate_video(payload: dict[str, Any], out_path: Optional[Path] = None, progress_cb: Optional[Callable[[str, int, str], None]] = None) -> Path:
    import torch
    params = validate_request(payload)
    if progress_cb:
        progress_cb("loading", 15, "Loading the open-source video model")
    pipe = load_pipeline()
    if progress_cb:
        progress_cb("generating", 35, "Generating frames on the GPU")
    generator = torch.Generator(device="cuda").manual_seed(params["seed"])
    kwargs: dict[str, Any] = {
        "prompt": params["prompt"],
        "num_frames": params["num_frames"],
        "height": params["height"],
        "width": params["width"],
        "guidance_scale": params["guidance_scale"],
        "num_inference_steps": params["steps"],
        "generator": generator,
    }
    if params["negative_prompt"]:
        kwargs["negative_prompt"] = params["negative_prompt"]
    try:
        with torch.inference_mode():
            result = pipe(**kwargs)
        frames = getattr(result, "frames", result)
        out = Path(out_path) if out_path else OUTPUT_DIR / "clip.mp4"
        _export_mp4(frames, out, params["fps"])
    except GenerationError:
        raise
    except torch.cuda.OutOfMemoryError as exc:
        _clear_cuda()
        raise GenerationError("The GPU ran out of memory. Use 8 frames, 256x256, and 10-15 steps, then Runtime -> Restart session.") from exc
    except Exception as exc:
        message = str(exc).lower()
        if "out of memory" in message or ("cuda" in message and "memory" in message):
            _clear_cuda()
            raise GenerationError("The GPU ran out of memory. Use 8 frames, 256x256, and 10-15 steps, then Runtime -> Restart session.") from exc
        raise GenerationError("Video generation failed on the Colab GPU.") from exc
    finally:
        _clear_cuda()
    if progress_cb:
        progress_cb("completed", 100, "MP4 ready")
    return out


def generate_video_with_fallback(payload: dict[str, Any], out_path: Optional[Path] = None) -> tuple[Path, dict[str, Any]]:
    try:
        path = generate_video(payload, out_path=out_path)
        return path, {"retried": False, "settings": validate_request(payload)}
    except GenerationError as exc:
        if "memory" not in str(exc).lower():
            raise
        print("OOM on first try; retrying 8 frames / 10 steps")
        _clear_cuda()
        reduced = dict(payload)
        reduced.update({"num_frames": 8, "steps": 10, "height": 256, "width": 256})
        path = generate_video(reduced, out_path=out_path)
        return path, {"retried": True, "settings": validate_request(reduced), "reason": str(exc)}


def model_info() -> dict[str, Any]:
    report = gpu_report()
    report["model_id"] = _model_id
    report["optimizations"] = _device_note
    report["loaded"] = _pipe is not None
    return report


def friendly_error(exc: BaseException) -> str:
    if isinstance(exc, GenerationError):
        return str(exc)
    text = str(exc).lower()
    if "out of memory" in text:
        return "The GPU ran out of memory. Use 8 frames and 256x256, then restart the Colab runtime."
    if "cuda" in text and "not" in text:
        return "GPU unavailable. In Colab choose Runtime -> Change runtime type -> GPU."
    return "Video generation failed. Keep the Colab notebook running and try again."
