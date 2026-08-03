from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.core.urgency import UrgencyLevel


class PredictionRequest(BaseModel):
    """DTO de entrada para a predição de um laudo médico."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Texto do laudo médico a ser classificado.",
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        """Rejeita textos compostos apenas por espaços em branco."""
        if not value.strip():
            raise ValueError("O laudo não pode ser um texto em branco.")
        return value


class PredictionResponse(BaseModel):
    """DTO de saída da predição de um laudo médico."""

    model_config = ConfigDict(extra="forbid")

    level: UrgencyLevel
    label: str
    confidence: float
    latency_ms: float
