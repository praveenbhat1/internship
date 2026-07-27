# PAR.vision — showcase website

A premium single-page site for the Pedestrian Attribute Recognition project, with a **real
working live demo**: upload an image and the trained model detects 23 attributes, exactly like
`demo_full.py`.

Palette: black / white / olive. Stack: Vite + React + Tailwind, GSAP + Lenis, Three.js hero.

## Sections
1. **Home** — hero with rotating 3D object
2. **Live demo** — upload an image → real model output (attributes, CMAA heatmaps, DACG, gender abstention)
3. **Results** — headline metrics + cross-dataset (PA-100K → PETA)
4. **What we used** — SigLIP-2, OCFR, CMAA, DACG, CCLoss + datasets

## Running it (needs BOTH servers)

The website is static, but the model runs in a small Python backend. Start both:

**1. Model backend** (loads the checkpoint, ~10s):
```bash
cd ../mvp
python serve.py            # http://127.0.0.1:8000
```

**2. Website:**
```bash
npm install                # first time only
npm run dev                # http://localhost:5173
```

Open the site, scroll to **Live demo**, upload a cropped photo of one person (or click a sample).
If the demo says "server isn't running", start `serve.py` and reload.

## Deploying
- Frontend: `npm run build` → deploy `dist/` (Vercel / Netlify).
- Backend: the model needs a Python host (Render / a small GPU or CPU box). Point the frontend at
  it by setting `VITE_API=https://your-backend` before `npm run build`.
