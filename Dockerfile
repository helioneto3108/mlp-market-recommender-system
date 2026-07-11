# syntax=docker/dockerfile:1

FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project


FROM python:3.12-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app \
    && useradd --system --gid app app \
    && mkdir -p /app \
    && chown app:app /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv

COPY --chown=app:app src/ ./src/
COPY --chown=app:app scripts/ ./scripts/
COPY --chown=app:app params.yaml dvc.yaml dvc.lock ./
COPY --chown=app:app .dvc/config ./.dvc/config
COPY --chown=app:app .dvcignore ./.dvcignore

USER app

CMD ["python", "-m", "scripts.predict", "--help"]