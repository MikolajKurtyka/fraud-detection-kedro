# Scoring API image. Multi-stage: dependencies are resolved with uv in a builder
# stage, the runtime stage only gets the virtualenv and source code.

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

FROM python:3.12-slim
# libgomp1 is required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home app
WORKDIR /app
COPY --from=builder --chown=app:app /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
# One BLAS/OpenMP thread per worker: single-row scoring gains nothing from more, and
# several workers would otherwise oversubscribe the CPUs. Scale with WEB_CONCURRENCY.
ENV OMP_NUM_THREADS=1 WEB_CONCURRENCY=2 PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus
USER app
EXPOSE 8000
# Liveness only: the API stays up (and healthy) without a model; /health/ready says why.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live')"
# Shell form so WEB_CONCURRENCY expands; the metrics dir is reset so /metrics aggregates
# only the current workers. Request logging is done by the app middleware.
CMD rm -rf "$PROMETHEUS_MULTIPROC_DIR" && mkdir -p "$PROMETHEUS_MULTIPROC_DIR" \
    && exec uvicorn fraud_detection.api.main:app --host 0.0.0.0 --port 8000 \
    --workers "${WEB_CONCURRENCY:-2}" --no-access-log
