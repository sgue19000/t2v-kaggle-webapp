# Kaggle GPU backend

Temporary single-user API. Not a production server.

## Model

Diffusers `TextToVideoSDPipeline`
- `ali-vilab/text-to-video-ms-1.7b`
- fallback `damo-vilab/text-to-video-ms-1.7b` (ID used in official HF snippets)

FP16 + VAE slicing/tiling + CPU offload. Defaults: 16 frames, 256x256, 8 fps, 20 steps, guidance 9.

## API

- GET /health
- POST /generate
- GET /status/{job_id}
- GET /video/{job_id}

One GPU job at a time. CORS allows https://sgue19000.github.io and localhost.
