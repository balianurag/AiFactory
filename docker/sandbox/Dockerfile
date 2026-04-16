# ─────────────────────────────────────────────────────────────────────────────
# AI Factory — Sandbox Image
# Multi-stage build: builder installs deps, runtime is minimal.
#
# Build:  docker build -t aifactory-sandbox:latest docker/sandbox/
# Test:   docker run --rm aifactory-sandbox:latest python -c "import numpy; print(numpy.__version__)"
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ──────────────────────────────────────────────
FROM python:3.12-slim AS builder

# ── Step 1: copy Zscaler cert and activate it ─────────────────────────────────
# ca-certificates is tiny — plain HTTP to deb.debian.org works fine here.
COPY docker/sandbox/certs/zscaler.crt /usr/local/share/ca-certificates/zscaler.crt
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

# ── Step 2: switch apt sources to HTTPS (Zscaler handles HTTPS cleanly) ──────
RUN sed -i \
    -e 's|http://deb.debian.org|https://deb.debian.org|g' \
    -e 's|http://security.debian.org|https://security.debian.org|g' \
    /etc/apt/sources.list.d/debian.sources

# ── Step 3: install minimal build tools ──────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (uid=1000)
RUN groupadd -r sandbox --gid=1000 && \
    useradd -r -g sandbox --uid=1000 --home-dir=/workspace --shell=/bin/bash sandbox

WORKDIR /build

# Trust the cert for pip / requests
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt

# --prefer-binary: use pre-built wheels, avoiding gcc/g++ entirely.
# All ARM64/x86-64 packages in requirements.txt have pre-built wheels on PyPI.
COPY docker/sandbox/requirements.txt .
RUN pip install --no-cache-dir --prefer-binary --prefix=/install -r requirements.txt


# ── Stage 2: runtime image ───────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# ── Step 1: copy Zscaler cert and activate it ─────────────────────────────────
COPY docker/sandbox/certs/zscaler.crt /usr/local/share/ca-certificates/zscaler.crt
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

# ── Step 2: switch apt sources to HTTPS ──────────────────────────────────────
RUN sed -i \
    -e 's|http://deb.debian.org|https://deb.debian.org|g' \
    -e 's|http://security.debian.org|https://security.debian.org|g' \
    /etc/apt/sources.list.d/debian.sources

# ── Runtime system packages (curl, git, sqlite3, jq, node, npm) ──────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        bash \
        curl \
        wget \
        git \
        jq \
        make \
        ca-certificates \
        sqlite3 \
        libgomp1 \
        libglib2.0-0 \
        libssl3 \
        libffi8 \
        nodejs \
        npm \
    && rm -rf /var/lib/apt/lists/* \
    && node --version \
    && npm --version

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Create sandbox user & workspace
RUN groupadd -r sandbox --gid=1000 && \
    useradd -r -g sandbox --uid=1000 --home-dir=/workspace --shell=/bin/bash sandbox && \
    mkdir -p /workspace && \
    chown sandbox:sandbox /workspace

# Verify critical imports
RUN python -c "\
import numpy, pandas, matplotlib, scipy, sklearn, requests, httpx, \
pydantic, yaml, json, sqlite3, re, os, sys, pathlib, asyncio, \
subprocess, hashlib, base64, datetime, collections, itertools, \
functools, typing, dataclasses; \
print('All critical packages imported successfully')"

# ── Security hardening ───────────────────────────────────────────────────────
RUN find / -perm /6000 -type f -exec chmod a-s {} \; 2>/dev/null || true
RUN chmod o-w /tmp /var/tmp 2>/dev/null || true

USER sandbox
WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONUNBUFFERED=1 \
    HOME=/workspace \
    PATH="/usr/local/bin:$PATH" \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

CMD ["sleep", "infinity"]

LABEL org.opencontainers.image.title="AI Factory Sandbox" \
      org.opencontainers.image.description="Isolated Python execution environment for AI Factory workers" \
      org.opencontainers.image.vendor="AI Factory" \
      aifactory.sandbox.version="1.0.0"
