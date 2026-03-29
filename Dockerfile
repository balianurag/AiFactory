# ─────────────────────────────────────────────────────────────────────────────
# AiFactory Backend — Dockerfile
# Python 3.12 · uv · FastAPI / Uvicorn
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

# Pull uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# System dependencies (curl for healthcheck, build tools for some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies (cached layer — no project source needed) ─────
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# ── Copy source code then install the project itself ──────────────────────────
COPY src/ ./src/
RUN uv sync --frozen --no-dev

# ── Runtime environment ───────────────────────────────────────────────────────
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# ── Start server ──────────────────────────────────────────────────────────────
CMD ["uvicorn", "aifactory.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
