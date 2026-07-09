# ---------------------------------------------------------------------------
# Immo Eliza — FastAPI prediction API
# ---------------------------------------------------------------------------
# Single-stage image serving the tuned-XGBoost sale & rent models plus the
# neighbourhood-aware endpoints (explain / similar / priciness / geocode /
# invest). Those endpoints import the repo-root packages (geo/, invest/) and
# read the data/ artifacts, so the image is built from the REPO ROOT and mirrors
# the repo layout under /app (predict.py resolves the repo root as two levels up
# from itself, i.e. /app).
#
#   Build (from the repo root):  docker build -f api/Dockerfile -t immo-api .
#   Run (host 8010 -> container 8000):  docker run -p 8010:8000 immo-api
# On Render this is built automatically (see ../render.yaml, dockerContext: .).
# ---------------------------------------------------------------------------
FROM python:3.13-slim

# XGBoost needs the OpenMP runtime (libgomp) at runtime; curl for the HEALTHCHECK.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Install deps first so Docker caches this layer across code changes. XGBoost 3.x
# pulls in nvidia-nccl-cu12 (~300 MB) for optional GPU use; Render's free tier is
# CPU-only, so we drop it to keep the image lean (XGBoost falls back to CPU).
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y nvidia-nccl-cu12 2>/dev/null || true

# Application code + cross-cutting packages, mirroring the repo layout so the
# repo-root imports (`from geo import ...`, `from invest import ...`) resolve.
COPY api/    /app/api/
COPY geo/    /app/geo/
COPY invest/ /app/invest/
# Compact runtime data artifacts the endpoints read (the heavy raw/cache
# partitions are excluded via the repo-root .dockerignore).
COPY data/geo/      /app/data/geo/
COPY data/invest/   /app/data/invest/
COPY data/listings/ /app/data/listings/

WORKDIR /app/api
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/" || exit 1

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
