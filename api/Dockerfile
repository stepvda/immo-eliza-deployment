# ---------------------------------------------------------------------------
# Immo Eliza — FastAPI prediction API
# ---------------------------------------------------------------------------
# Slim, single-stage image that serves the tuned-XGBoost sale & rent models.
# Build context is the api/ folder:   docker build -t immo-api api/
# Run (host localhost:8010 -> container 8000):  docker run -p 8010:8000 immo-api
# On Render, this Dockerfile is built automatically (see ../render.yaml).
# ---------------------------------------------------------------------------
FROM python:3.13-slim

# XGBoost needs the OpenMP runtime (libgomp) at runtime; curl is handy for the
# container HEALTHCHECK. Clean apt lists to keep the image small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Install dependencies first so Docker caches this layer across code changes.
# XGBoost 3.x pulls in nvidia-nccl-cu12 (~300 MB) for optional GPU acceleration;
# Render's free tier is CPU-only, so we uninstall it to keep the image lean
# (XGBoost falls back to CPU without it). This is the "remove unnecessary
# dependencies" step the deployment brief asks for.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y nvidia-nccl-cu12 2>/dev/null || true

# Copy the application code and the model artifacts.
COPY . .

EXPOSE 8000

# Render provides $PORT; default to 8000 locally. Use shell form so $PORT expands.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/" || exit 1

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
