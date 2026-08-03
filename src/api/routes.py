from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Response

from src.api.dependencies import get_inference, get_metrics
from src.api.metrics import PrometheusMetrics
from src.domain.interfaces import ModelInference
from src.domain.schemas import PredictionRequest, PredictionResponse

router = APIRouter()


@router.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    """Healthcheck simples da API."""
    return {"status": "ok"}


@router.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["prediction"],
    responses={200: {"model": PredictionResponse}},
)
async def predict(
    payload: PredictionRequest,
    inference: Annotated[ModelInference, Depends(get_inference)],
    metrics: Annotated[PrometheusMetrics, Depends(get_metrics)],
) -> PredictionResponse:
    """Classifica um laudo médico e retorna a categoria de urgência."""
    import time

    start = time.perf_counter()
    level, confidence = inference.predict(payload.text)
    latency_ms = (time.perf_counter() - start) * 1000.0
    metrics.observe_prediction(level.value)
    return PredictionResponse(
        level=level,
        label=level.value,
        confidence=round(confidence, 4),
        latency_ms=round(latency_ms, 4),
    )


@router.get("/metrics", include_in_schema=False)
async def metrics_endpoint(metrics: Annotated[PrometheusMetrics, Depends(get_metrics)]) -> Response:
    """Expõe as métricas no formato do Prometheus."""
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)
