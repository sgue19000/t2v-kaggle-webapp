# Lumen Clip

Mobile-first text-to-video web app.

- Frontend: Next.js on GitHub Pages
- Backend: temporary FastAPI process in a free Kaggle GPU notebook
- Model: [ali-vilab/text-to-video-ms-1.7b](https://huggingface.co/ali-vilab/text-to-video-ms-1.7b) via Diffusers `TextToVideoSDPipeline` (fallback `damo-vilab/text-to-video-ms-1.7b`)

Repo: https://github.com/sgue19000/t2v-kaggle-webapp

## Step 2 procedure

1. Open https://www.kaggle.com and sign in.
2. Create a new notebook.
3. Settings → Accelerator → GPU P100 or T4 → Save → Restart session.
4. Upload and run `kaggle/video_generator.ipynb` top to bottom.
5. Wait for the model-load cell to finish.
6. Copy the printed trycloudflare URL:

```
========================================
PUBLIC API URL:
https://xxxx.trycloudflare.com
========================================
```

7. Open https://sgue19000.github.io/t2v-kaggle-webapp/
8. Paste the URL into the API URL field and tap Save API URL.
9. Click Generate Video.
10. Wait until status is completed.
11. Play or download the MP4.

Keep the Kaggle notebook running the entire time. If it disconnects, start again and paste the new URL. No rebuild is required.

## Local frontend

```bash
git clone https://github.com/sgue19000/t2v-kaggle-webapp.git
cd t2v-kaggle-webapp
npm install
npm run lint
npm run typecheck
npm run build
python3 -m py_compile kaggle/generator.py kaggle/server.py
```

P100 defaults: 16 frames, 256x256, 8 fps, 20 steps, guidance 9.
