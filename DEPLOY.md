# Deploying PAR.vision online

The site has two parts that deploy separately:

- **Frontend** (`website/`) — a static React build → host on **Vercel** (free).
- **Backend** (`mvp/serve.py`) — loads the model, needs real RAM → host on a
  **Hugging Face Space** (free CPU tier has 16 GB RAM; Render/Railway free tiers
  are too small for SigLIP-2).

The frontend finds the backend through the `VITE_API` environment variable.

---

## Step 1 — Deploy the backend to a Hugging Face Space

1. Create an account at https://huggingface.co, then **New Space** → SDK: **Docker** →
   name it e.g. `par-api`.
2. In that Space repo, add the contents of this project's `mvp/` folder **plus** the
   `mvp/Dockerfile` at the Space root. Easiest:
   ```bash
   git clone https://huggingface.co/spaces/<you>/par-api
   cp -r internship/mvp/* par-api/           # includes Dockerfile, serve.py, features/
   cd par-api && git add . && git commit -m "PAR API" && git push
   ```
3. The Space builds automatically. First boot downloads SigLIP-2 (~1.5 GB) — give it
   a few minutes. When it's green, your API URL is:
   ```
   https://<you>-par-api.hf.space
   ```
   Check it: open `https://<you>-par-api.hf.space/health` → should return `{"status":"ok"}`.

> CORS is already open (`allow_origins=["*"]`), so the Vercel site can call it.

## Step 2 — Deploy the frontend to Vercel

1. Create an account at https://vercel.com and **Add New → Project**, import the
   GitHub repo `praveenbhat1/internship`.
2. Set **Root Directory** = `website` (important — the site isn't at the repo root).
   Vercel auto-detects Vite; `vercel.json` handles the rest.
3. Under **Environment Variables**, add:
   ```
   VITE_API = https://<you>-par-api.hf.space
   ```
4. **Deploy.** You'll get a public URL like `https://par-vision.vercel.app`.

Done — anyone can open that URL and run the live demo.

---

## Updating after changes
- Push to `main` → Vercel redeploys the frontend automatically.
- Push to the Space repo → the backend rebuilds automatically.

## Local development (no deploy)
Run both servers locally instead — see `website/README.md`. Locally the frontend
defaults to `http://127.0.0.1:8000`, so no env var is needed.

## Notes / limits
- The free HF Space **sleeps** when idle and takes ~30 s to wake on the next request —
  fine for a portfolio demo. Keep the local instructions handy for a live presentation.
- If you'd rather not host the model, deploy only the frontend and demo the model
  locally during your presentation; the site shows a clear "server isn't running"
  message when the backend is unreachable.
