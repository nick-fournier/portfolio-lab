# Portfolio lab: one image for both the dashboard (`plab serve`) and the scheduler
# (`plab schedule`). Data lives in a volume mounted at /data.

FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first, so code changes don't invalidate this layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim
ARG GIT_SHA=""
RUN useradd --uid 1001 --user-group --no-create-home lab
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH \
    PORTFOLIO_DATA_DIR=/data \
    GIT_SHA=$GIT_SHA \
    PYTHONUNBUFFERED=1
USER lab
WORKDIR /data
EXPOSE 8100
HEALTHCHECK --interval=60s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8100/healthz')" || exit 1
CMD ["plab", "serve", "--host", "0.0.0.0", "--port", "8100"]
