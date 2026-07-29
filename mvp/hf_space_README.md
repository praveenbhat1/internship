---
title: PAR API
emoji: 🧍
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# PAR API

FastAPI backend that serves the trained Pedestrian Attribute Recognition model
(`serve.py`). Endpoints:

- `GET /health` — status
- `POST /predict` — image → 23 attributes + gender abstention + CMAA/DACG/feature images

First boot downloads the SigLIP-2 backbone (~1.5 GB), so the initial build takes
a few minutes.
