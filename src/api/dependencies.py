from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import Request

from src.api.metrics import PrometheusMetrics
from src.core.config import Settings
from src.domain.interfaces import ModelInference


@lru_cache
def get_settings() -> Settings:
    """Fornece as configurações globais (cached) para injeção de dependência."""
    return Settings()


@lru_cache
def get_metrics() -> PrometheusMetrics:
    """Fornece o exportador de métricas (singleton) para injeção de dependência."""
    return PrometheusMetrics()


def get_inference(request: Request) -> ModelInference:
    """Fornece o preditor ONNX armazenado no estado da aplicação."""
    settings = get_settings()
    onnx_path: Path = settings.onnx_model_path
    if not onnx_path.exists():
        raise FileNotFoundError(f"Modelo ONNX não encontrado em {onnx_path}.")
    inference: ModelInference = request.app.state.inference
    return inference
