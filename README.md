# Lumen Clip

GitHub Pages frontend → Google Colab free GPU → FastAPI → Cloudflare tunnel.

- Model: [ali-vilab/text-to-video-ms-1.7b](https://huggingface.co/ali-vilab/text-to-video-ms-1.7b) (`TextToVideoSDPipeline`)
- Site: https://sgue19000.github.io/t2v-kaggle-webapp/
- Colab notebook: https://colab.research.google.com/github/sgue19000/t2v-kaggle-webapp/blob/main/colab/video_generator_colab.ipynb

This is not a permanent server. The Colab runtime can disconnect. The tunnel URL changes every session. Generated files die with the runtime. GPU type varies (often T4, not guaranteed).

## Start backend (Android)

1. Open Google Colab.
2. Open `colab/video_generator_colab.ipynb` from the link above.
3. Runtime → Change runtime type → GPU.
4. Runtime → Run all.
5. Wait for the model to load.
6. Wait for the tiny test video.
7. Copy the printed PUBLIC API URL (`https://….trycloudflare.com`).
8. Open the GitHub Pages website.
9. Paste the URL into API URL.
10. Save.
11. Generate a video.

Keep the Colab tab open the whole time.

## Defaults

8 frames, 256x256, 8 fps, 15 steps, guidance 9. Server rejects larger resolutions.
