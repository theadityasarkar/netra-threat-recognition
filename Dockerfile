# ─────────────────────────────────────────────────────────────────────────────
# NETRA — production container
# SIH 2026 · PS 26189 · Ministry of Home Affairs · NCRB Women Safety Division
#
# Multi-stage build. The builder stage carries the compilers and wheel caches
# needed to resolve numpy/scikit-learn; the runtime stage copies only the
# finished virtualenv, so the shipped image contains no build toolchain.
#
#   docker build -t netra:latest .
#   docker run --rm -p 8501:8501 netra:latest
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Never write .pyc into the image; never buffer logs behind a container boundary.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# build-essential is needed only to compile any dependency without a wheel for
# this platform. It stays in the builder and never reaches the runtime image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# Self-contained virtualenv, so the runtime stage is a single COPY.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only the manifest first: this layer is cached and dependency resolution
# (the slow step) is skipped on every rebuild that changes application code only.
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="NETRA" \
      org.opencontainers.image.description="AI-Powered Criminal Network Analysis System — SIH 2026 PS 26189" \
      org.opencontainers.image.vendor="Ministry of Home Affairs · NCRB Women Safety Division" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="2.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    # Streamlit must bind 0.0.0.0 to be reachable from outside the container.
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    # Writable SQLite location; docker-compose mounts a volume here.
    NETRA_DB_PATH=/app/data/netra.db

# curl is kept only for the container HEALTHCHECK below.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Run unprivileged: a compromise in the web layer must not own the container.
RUN useradd --create-home --uid 10001 netra

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=netra:netra . .

# data/ must be writable by the app user for the SQLite case registry and exports.
RUN mkdir -p /app/data/exports /app/data/samples \
 && chown -R netra:netra /app/data

USER netra

EXPOSE 8501

# Streamlit's own readiness endpoint — the app is not "up" until this returns 200.
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD curl --fail --silent http://localhost:${PORT:-8501}/_stcore/health || exit 1

# ``sh -c`` is required so ${PORT} is expanded at runtime: Render and most free
# PaaS tiers inject the port to bind, and ignoring it makes the service unreachable.
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
