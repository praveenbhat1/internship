# Backend image for the PAR live-demo API (serve.py).
# Build context = this mvp/ folder.  Serves FastAPI on port 7860 (Hugging Face Spaces default).
FROM python:3.11-slim

ENV HF_HOME=/app/hf_cache \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/app/mpl_cache

WORKDIR /app

# system deps for pillow / pillow-heif
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo libpng16-16 && rm -rf /var/lib/apt/lists/*

# CPU-only torch first, so the CUDA wheels (~2.5 GB, useless on a CPU Space) are never pulled.
COPY requirements-space.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3" \
 && pip install --no-cache-dir -r requirements-space.txt

# Bake the SigLIP-2 backbone (~1.5 GB) into the image at BUILD time.
# Without this the first visitor triggers the download and their request hangs for minutes.
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('google/siglip2-large-patch16-256')"

COPY . .

# The backbone is cached above, so refuse to reach the network at runtime — a hung
# HF request can otherwise make /predict stall instead of failing fast.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Cloud Run injects $PORT (8080) and routes traffic there; locally it's unset, so
# fall back to 7860. Shell form is required for the variable to expand.
EXPOSE 8080
CMD ["sh", "-c", "uvicorn serve:app --host 0.0.0.0 --port ${PORT:-7860}"]
