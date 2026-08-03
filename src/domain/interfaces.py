from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.urgency import UrgencyLevel


class ModelInference(ABC):
    """Interface (ABC) para estratégias de inferência de modelos."""

    @abstractmethod
    def predict(self, text: str) -> tuple[UrgencyLevel, float]:
        """Classifica um laudo e retorna (nível de urgência, confiança)."""
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, text: str) -> dict[UrgencyLevel, float]:
        """Retorna as probabilidades de cada categoria de urgência."""
        raise NotImplementedError

    @abstractmethod
    def supports_batch(self) -> bool:
        """Indica se a estratégia suporta predição em lote (batch)."""
        raise NotImplementedError
