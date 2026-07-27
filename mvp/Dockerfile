# Backend image for the PAR live-demo API (serve.py).
# Build context = this mvp/ folder.  Serves FastAPI on port 7860 (Hugging Face Spaces default).
FROM python:3.11-slim

ENV HF_HOME=/app/hf_cache \
    PYTHONUNBUFFERED=1

WORKDIR /app

# system deps for pillow / pillow-heif
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo libpng16-16 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
# SigLIP-2 (~1.5 GB) downloads on first boot, then is cached in the image layer/volume.
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "7860"]
