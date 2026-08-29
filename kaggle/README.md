# Kaggle GPU backend

This folder is the inference side of Lumen Clip. It is **not** a permanent server.

Free Kaggle GPU notebooks last about 12 hours, disconnect if idle, and use a Tesla P100 or T4 (~16 GB VRAM).

## Model

`ali-vilab/text-to-video-ms-1.7b` (fallback `damo-vilab/text-to-video-ms-1.7b`)

1.7B `TextToVideoSDPipeline` with FP16, VAE slicing, and CPU offload. Newer models (Wan 14B, HunyuanVideo, CogVideoX-5B) do not fit a free Kaggle session reliably.

License: CC-BY-NC-4.0. English prompts only.

Recommended P100 settings: 16 frames, 256x256, 8 fps, 25 steps, guidance 9.

## API

- GET /health
- POST /generate
- GET /status/{job_id}
- GET /video/{job_id}
