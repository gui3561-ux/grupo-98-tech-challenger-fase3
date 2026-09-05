# Stage 1: instalação de dependências
FROM python:3.11-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.11.0 /uv /uvx /bin/

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Stage 2: imagem final da API
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

COPY --from=base /app/.venv /app/.venv

COPY src ./src
COPY metrics ./metrics
COPY scripts/entrypoint.sh ./scripts/entrypoint.sh
RUN chmod +x scripts/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
