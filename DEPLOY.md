# Deploying PAR.vision online (free)

The site has two parts that deploy separately:

- **Frontend** (`website/`) — a static React build → **Vercel** (free forever, no sleep).
- **Backend** (`mvp/serve.py`) — loads SigLIP-2, peaks at ~3-4 GB RAM → **Google Cloud Run**
  (scales to zero; the always-free tier covers a demo). Render, Railway and Fly free
  instances cap at 512 MB, which is far too small.

> **Hugging Face Spaces no longer works for this.** Around **8 July 2026** HF began
> requiring a PRO subscription to host Gradio or Docker Spaces on free `cpu-basic`:
> *"Static Spaces are free for everyone, but hosting Gradio and Docker Spaces on free
> cpu-basic requires a PRO subscription."* The old Space instructions are kept at the
> bottom of this file for reference only.

The frontend finds the backend through the `VITE_API` environment variable.

---

## Step 1 — Deploy the backend to Google Cloud Run

Cloud Run builds the `mvp/Dockerfile` for you — no local Docker install, and no Git LFS
(the 24 MB `par_full.pt` is just a file in the build context).

1. Create a project at https://console.cloud.google.com and enable billing. The card is
   for identity; the always-free tier below is permanent and separate from the $300 trial.

2. Install the CLI and authenticate:

   ```bash
   brew install --cask google-cloud-sdk
   gcloud auth login
   gcloud config set project <your-project-id>
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
   ```

3. Deploy straight from `mvp/` — this uploads the build context, builds the image with
   Cloud Build, and rolls it out:

   ```bash
   cd ~/internship/mvp
   gcloud run deploy par-api \
     --source . \
     --region asia-south1 \
     --memory 4Gi \
     --cpu 2 \
     --cpu-boost \
     --timeout 300 \
     --min-instances 0 \
     --max-instances 1 \
     --allow-unauthenticated
   ```

   The flags that matter:
   - `--memory 4Gi` — `serve.py` loads the full SigLIP-2 (vision + text towers) in fp32
     before narrowing to the vision tower. 2 GiB OOMs during startup.
   - `--cpu-boost` — full CPU during cold start, so the model loads well inside Cloud
     Run's 240 s startup budget. `serve.py` loads at import time, so uvicorn does not
     bind the port until the model is ready.
   - `--min-instances 0` — scales to zero. **Setting this to 1 bills 24/7 and is the
     one easy way to turn this into a real monthly charge.**
   - `--max-instances 1` — caps the blast radius if the URL gets crawled.

4. The first build takes **~10-15 minutes** (it bakes the 1.5 GB backbone into the
   image). When it finishes, gcloud prints the service URL. Check it:

   ```
   https://par-api-xxxxxxxx.asia-south1.run.app/health   →  {"status":"ok"}
   ```

> CORS is already open (`allow_origins=["*"]`), so the Vercel site can call it.

## Step 2 — Deploy the frontend to Vercel

1. Create an account at https://vercel.com and **Add New → Project**, import the
   GitHub repo `praveenbhat1/internship`.
2. Set **Root Directory** = `website` (important — the site isn't at the repo root).
   Vercel auto-detects Vite; `vercel.json` handles the rest.
3. Under **Environment Variables**, add — note **no trailing slash**:
   ```
   VITE_API = https://par-api-xxxxxxxx.asia-south1.run.app
   ```
   This is baked in at build time, so if you change it later you must **redeploy**,
   not just restart.
4. **Deploy.** You'll get a public URL like `https://par-vision.vercel.app`.

## Step 3 — Cold starts

With `--min-instances 0` the container is torn down when idle, so the first request
after a quiet spell waits ~40-60 s while the model loads. The site already retries
`/health` for 75 s before showing "offline", so this degrades gracefully.

Do **not** "fix" this with `--min-instances 1` — that keeps an instance warm around the
clock and blows through the free tier. Before a live presentation, just open `/health`
yourself ~2 minutes early so the first demo request is warm.

## What this actually costs

| Resource | Always-free allowance | This project's use |
|---|---|---|
| Cloud Run CPU | 180,000 vCPU-s/month | far under — CPU is billed only while a request runs |
| Cloud Run memory | 360,000 GiB-s/month | at 4 GiB that's ~25 h of active time/month |
| Cloud Run requests | 2,000,000/month | far under |
| Artifact Registry | **0.5 GB** storage | **the image is ~3-4 GB — this is the one overage** |

Compute is genuinely $0 for a portfolio demo. The honest exception is image storage:
Artifact Registry bills ~$0.10/GB/month past 0.5 GB, so expect **roughly $0.30-0.40 a
month** (about ₹30). The $300 trial credit absorbs it for the first 90 days. To keep it
minimal, delete superseded images so only the current one is stored:

```bash
gcloud artifacts docker images list \
  asia-south1-docker.pkg.dev/<project>/cloud-run-source-deploy/par-api
gcloud artifacts docker images delete <digest> --delete-tags
```

Set a budget alert at $1 (Billing → Budgets & alerts) if you want a hard tripwire.

---

## Why the build is set up this way (glitches this avoids)

| Problem | Fix in this repo |
|---|---|
| `pip install torch` pulls ~2.5 GB of CUDA wheels a CPU-only host can't use — slow, fragile builds | `Dockerfile` installs from the **CPU wheel index**; `requirements-space.txt` drops Gradio/SciPy |
| First visitor triggers a 1.5 GB model download and their request hangs for minutes | backbone is `snapshot_download`ed at **build** time and baked into the image |
| A hung HuggingFace call at runtime stalls `/predict` | `HF_HUB_OFFLINE=1` set after the model is baked in — fails fast instead |
| Cloud Run routes to `$PORT`, not the hardcoded 7860 | `CMD` uses `${PORT:-7860}` in shell form — works on Cloud Run and locally |
| A cold-started backend shows a false "server isn't running" banner | frontend retries `/health` 15× over 75 s; `/predict` has a 120 s timeout with a clear message |
| Matplotlib tries to write a font cache to a read-only home dir | `MPLCONFIGDIR` set in the Dockerfile |

## Updating after changes
- Push to `main` → Vercel redeploys the frontend automatically.
- Backend changes are **not** automatic. Re-run the same `gcloud run deploy` command
  from `mvp/`; subsequent builds are much faster because the backbone layer is cached.

## Local development (no deploy)
Run both servers locally instead — see `website/README.md`. Locally the frontend
defaults to `http://127.0.0.1:8000`, so no env var is needed.

## Cost
See "What this actually costs" above: Vercel and Cloud Run compute are $0, with roughly
$0.30-0.40/month for Artifact Registry image storage. Vercel's limit is 100 GB/month of
bandwidth, which a portfolio demo will not hit.

## Superseded: the Hugging Face Space route
Until July 2026 the backend ran on a free HF Docker Space (2 vCPU / 16 GB). That tier
now requires PRO. If you ever have a PRO account, the old flow was: create a Docker
Space, `git lfs track "*.pt"` **before** the first commit (HF rejects the 24 MB
`par_full.pt` over plain git), copy `Dockerfile`, `.dockerignore`,
`requirements-space.txt`, `serve.py`, `train_par_full.py`, `features/par_full.pt` and
the three `features/*.json` files to the repo root, and push. `hf_space_README.md`
holds the YAML header that Space needs.
