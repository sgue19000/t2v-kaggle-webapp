# Lumen Clip

Mobile-first text-to-video web app.

- Frontend: Next.js + React + TypeScript + Tailwind, hosted from this GitHub repository
- Backend: FastAPI inside a **temporary** free Kaggle GPU notebook
- Model: [ali-vilab/text-to-video-ms-1.7b](https://huggingface.co/ali-vilab/text-to-video-ms-1.7b) (fallback `damo-vilab/text-to-video-ms-1.7b`)

No OpenAI, Replicate, RunPod, Fal, or Stability APIs. GitHub Actions does **not** provide a GPU.

Repo: https://github.com/sgue19000/t2v-kaggle-webapp

## 1. Clone (Termux or desktop)

```bash
pkg update -y
pkg install -y git nodejs-lts python
git clone https://github.com/sgue19000/t2v-kaggle-webapp.git
cd t2v-kaggle-webapp
```

## 2. Install the web app

```bash
npm install
```

## 3. Configure the API URL

```bash
cp .env.example .env.local
```

After the Kaggle tunnel starts:

```
NEXT_PUBLIC_API_URL=https://YOUR-TUNNEL.trycloudflare.com
```

You can also paste that URL into the site API field. It is saved in the browser. No secrets belong here.

## 4. Create a Kaggle notebook

1. Sign in at https://www.kaggle.com
2. Create a new notebook
3. File → Upload notebook → `kaggle/video_generator.ipynb`

## 5. Enable GPU

Notebook → Settings → Accelerator → GPU P100 or GPU T4 → Save → Restart session.

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
```

## 6. Run the Kaggle backend

Run cells 1-6 top to bottom. Cell 6 prints `PUBLIC_API_URL=https://....trycloudflare.com`.

## 7. Connect the web app

Paste the HTTPS URL into the site API field or `.env.local`, then:

```bash
npm run dev
```

Open the printed local URL on your Android browser.

## 8. Generate the first video

Keep Kaggle running. Use 16 frames / 256px / 8 fps. Tap Generate Video. Wait 2-6 minutes. Preview and download.

## 9. Troubleshoot

- CUDA false → enable GPU and restart
- Out of memory → 16 frames, 256px, 20 steps
- API offline → re-run tunnel cell, paste the new URL
- Mixed content → URL must be https
- HTTP 409 → one job at a time

## 10. Limits you cannot avoid on free Kaggle GPU

Not a production server. ~12 hour cap. Idle disconnects. ~16 GB VRAM. Tunnel URL changes every session. 256px short clips. CC-BY-NC-4.0 model license.

## 11. P100 recommended settings

16 frames, 256x256, 8 fps, 20-25 steps, guidance 9.

## 12. Deploy the frontend

Repo → Settings → Pages → GitHub Actions.
Site: https://sgue19000.github.io/t2v-kaggle-webapp/

Paste the live tunnel URL in the page API field after deploy.

```bash
npm install
npm run lint
npm run typecheck
npm run build
python3 -m py_compile kaggle/generator.py kaggle/server.py
```
