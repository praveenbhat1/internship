# Deploying PAR.vision online (free)

The site has two parts that deploy separately:

- **Frontend** (`website/`) — a static React build → **Vercel** (free forever, no sleep).
- **Backend** (`mvp/serve.py`) — loads the model, needs real RAM → **Hugging Face Space**
  (free CPU tier = 2 vCPU / 16 GB RAM). Render and Railway free tiers cap at 512 MB,
  which is far too small for SigLIP-2.

The frontend finds the backend through the `VITE_API` environment variable.

---

## Step 1 — Deploy the backend to a Hugging Face Space

1. Create an account at https://huggingface.co, then **New Space** →
   SDK: **Docker** → **Blank** template → name it `par-api` → **Public**.

2. Clone it and copy **only the files the server needs** (don't copy all of `mvp/` —
   the label `.npy` files and figures are ~90 MB of dead weight):

   ```bash
   git clone https://huggingface.co/spaces/<you>/par-api
   cd par-api

   # Git LFS is REQUIRED: par_full.pt is 23 MB and HF rejects >10 MB files over plain git.
   git lfs install
   git lfs track "*.pt"

   mkdir -p features
   cp ~/internship/mvp/Dockerfile               .
   cp ~/internship/mvp/.dockerignore            .
   cp ~/internship/mvp/requirements-space.txt   .
   cp ~/internship/mvp/serve.py                 .
   cp ~/internship/mvp/train_par_full.py        .   # serve.py imports FullPAR from it
   cp ~/internship/mvp/features/par_full.pt     features/
   cp ~/internship/mvp/features/attributes.json features/
   cp ~/internship/mvp/features/thresholds.json features/
   cp ~/internship/mvp/features/metrics.json    features/
   cp ~/internship/mvp/hf_space_README.md       README.md   # the YAML header configures the Space

   git add -A && git commit -m "PAR API" && git push
   ```

3. The Space builds automatically. **The first build takes ~10-15 minutes** because the
   Dockerfile bakes the SigLIP-2 backbone (~1.5 GB) into the image. That is deliberate —
   it means no visitor ever waits for a model download.

4. When the badge turns green, check it:
   ```
   https://<you>-par-api.hf.space/health     →  {"status":"ok"}
   ```

> CORS is already open (`allow_origins=["*"]`), so the Vercel site can call it.

## Step 2 — Deploy the frontend to Vercel

1. Create an account at https://vercel.com and **Add New → Project**, import the
   GitHub repo `praveenbhat1/internship`.
2. Set **Root Directory** = `website` (important — the site isn't at the repo root).
   Vercel auto-detects Vite; `vercel.json` handles the rest.
3. Under **Environment Variables**, add — note **no trailing slash**:
   ```
   VITE_API = https://<you>-par-api.hf.space
   ```
   This is baked in at build time, so if you change it later you must **redeploy**,
   not just restart.
4. **Deploy.** You'll get a public URL like `https://par-vision.vercel.app`.

## Step 3 — Keep the Space awake

A free Space **pauses after 48 h of inactivity** and takes ~30-60 s to wake. The site
already retries `/health` for 75 s before showing "offline", so a cold start degrades
gracefully rather than looking broken. To avoid it entirely, add a free uptime ping:

- https://cron-job.org (free) → new job → URL `https://<you>-par-api.hf.space/health`,
  interval **every 6 hours**. That's enough to reset the 48 h idle timer.

Before a live presentation, open `/health` yourself ~2 minutes early so the first
demo request is warm.

---

## Why the build is set up this way (glitches this avoids)

| Problem | Fix in this repo |
|---|---|
| `pip install torch` pulls ~2.5 GB of CUDA wheels that a CPU Space can't use — slow, fragile builds | `Dockerfile` installs from the **CPU wheel index**; `requirements-space.txt` drops Gradio/SciPy |
| First visitor triggers a 1.5 GB model download and their request hangs for minutes | backbone is `snapshot_download`ed at **build** time and baked into the image |
| A hung HuggingFace call at runtime stalls `/predict` | `HF_HUB_OFFLINE=1` set after the model is baked in — fails fast instead |
| `par_full.pt` (23 MB) rejected by HF's 10 MB plain-git limit | `git lfs track "*.pt"` before the first commit |
| Sleeping Space shows a false "server isn't running" banner | frontend retries `/health` 15× over 75 s; `/predict` has a 120 s timeout with a clear message |
| Matplotlib tries to write a font cache to a read-only home dir | `MPLCONFIGDIR` set in the Dockerfile |

## Updating after changes
- Push to `main` → Vercel redeploys the frontend automatically.
- Push to the Space repo → the backend rebuilds automatically (subsequent builds are
  much faster; the backbone layer is cached).

## Local development (no deploy)
Run both servers locally instead — see `website/README.md`. Locally the frontend
defaults to `http://127.0.0.1:8000`, so no env var is needed.

## Cost
Everything above is **$0/month**, permanently — no credit card, no trial. Limits are
Vercel's 100 GB/month bandwidth and the Space's 48 h idle pause, neither of which a
portfolio demo will hit.
