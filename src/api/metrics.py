from __future__ import annotations

from collections.abc import Awaitable, Callable

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.requests import Request
from starlette.responses import Response

from src.domain.metrics_protocol import MetricsExporter

BUCKETS_MS: tuple[float, ...] = (1, 5, 10, 25, 50, 100, 250, 500, 1000)


class PrometheusMetrics(MetricsExporter):
    """Implementa as métricas do Prometheus para a API de predição."""

    def __init__(self) -> None:
        self._requests = Counter(
            "urgency_http_requests_total",
            "Total de requisições HTTP por status",
            ["status"],
        )
        self._latency = Histogram(
            "urgency_request_latency_ms",
            "Latência das requisições em milissegundos",
            buckets=BUCKETS_MS,
        )
        self._predictions = Counter(
            "urgency_predictions_total",
            "Total de predições por categoria de urgência",
            ["level"],
        )

    def observe_request(self, status_code: int, latency_ms: float) -> None:
        """Registra o status HTTP e a latência de uma requisição."""
        self._requests.labels(str(status_code)).inc()
        self._latency.observe(latency_ms)

    def observe_prediction(self, level: str) -> None:
        """Incrementa o contador da categoria de urgência prevista."""
        self._predictions.labels(level).inc()

    def render(self) -> tuple[bytes, str]:
        """Gera o corpo de métricas no formato Prometheus."""
        return generate_latest(), CONTENT_TYPE_LATEST

    async def middleware(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Middleware que mede latência e status de toda requisição HTTP."""
        import time

        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000.0
        self.observe_request(response.status_code, latency_ms)
        return response
