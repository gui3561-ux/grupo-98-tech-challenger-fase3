from __future__ import annotations

from typing import Protocol


class MetricsExporter(Protocol):
    """Contrato (Protocol) para a exportação de métricas do Prometheus."""

    def observe_request(self, status_code: int, latency_ms: float) -> None:
        """Registra a latência e o status HTTP de uma requisição."""
        ...

    def observe_prediction(self, level: str) -> None:
        """Registra a predição de uma determinada categoria de urgência."""
        ...
