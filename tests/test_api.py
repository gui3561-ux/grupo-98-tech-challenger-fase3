from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.core.config import Settings


@pytest.fixture()
def client(tmp_path) -> TestClient:
    settings = Settings(
        model_dir=tmp_path / "models",
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
    )
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier

    texts = [
        "exame normal sem achados",
        "achado grave emergência internação",
        "quadro leve recomenda acompanhamento",
    ]
    labels = ["normal", "urgente", "atencao"]
    pipeline = Pipeline(
        [("tfidf", TfidfVectorizer()), ("clf", DecisionTreeClassifier())]
    )
    pipeline.fit(texts, labels)

    from src.services.onnx_converter import ONNXConverter

    ONNXConverter.convert(pipeline, settings.onnx_model_path)

    from src.api.dependencies import get_metrics
    from src.api.main import create_app
    from src.infrastructure.predictors import ONNXInference

    app = create_app()
    app.state.settings = settings
    app.state.metrics = get_metrics()
    app.state.inference = ONNXInference(settings.onnx_model_path)
    return TestClient(app)


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_response_model(client: TestClient) -> None:
    response = client.post("/predict", json={"text": "achado grave emergência"})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in {"normal", "atencao", "urgente"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["latency_ms"] >= 0


def test_predict_rejects_blank_text(client: TestClient) -> None:
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 422


def test_metrics_exposes_prometheus_format(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "# HELP" in response.text
    assert "urgency_http_requests_total" in response.text
    assert "urgency_request_latency_ms" in response.text
    assert "urgency_predictions_total" in response.text


def test_middleware_records_request_metrics(client: TestClient) -> None:
    client.get("/health")
    client.post("/predict", json={"text": "achado grave emergência"})
    metrics_text = client.get("/metrics").text
    assert 'urgency_http_requests_total{status="200"}' in metrics_text
    assert "urgency_request_latency_ms_count" in metrics_text
    assert 'urgency_predictions_total{level="urgente"}' in metrics_text
