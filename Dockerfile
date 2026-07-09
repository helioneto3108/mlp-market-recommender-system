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

RUN groupadd --system app && useradd --system --gid app app

COPY --from=builder /app/.venv /app/.venv

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY params.yaml dvc.yaml ./
COPY .dvc/config ./.dvc/config
COPY .dvcignore ./.dvcignore

RUN chown -R app:app /app

USER app

CMD ["python", "-m", "scripts.predict", "--help"]