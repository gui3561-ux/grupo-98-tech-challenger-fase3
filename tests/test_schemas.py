from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core.urgency import UrgencyLevel
from src.domain.schemas import PredictionRequest, PredictionResponse


def test_prediction_request_strips_whitespace() -> None:
    req = PredictionRequest(text="  laudo de exemplo  ")
    assert req.text == "laudo de exemplo"


def test_prediction_request_rejects_blank() -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(text="   ")


def test_prediction_request_rejects_unknown_extra_field() -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(text="ok", extra="forbid")  # type: ignore[call-arg]


def test_prediction_response_accepts_enum_level() -> None:
    resp = PredictionResponse(
        level=UrgencyLevel.URGENTE,
        label="urgente",
        confidence=0.95,
        latency_ms=2.5,
    )
    assert resp.level == UrgencyLevel.URGENTE
