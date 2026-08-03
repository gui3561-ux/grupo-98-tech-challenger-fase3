from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from src.core.urgency import KEYWORDS_ATENCAO, KEYWORDS_NORMAL, KEYWORDS_URGENTE, UrgencyLevel


class SyntheticDataGenerator:
    """Gera um dataset sintético de laudos médicos com alvos de urgência."""

    _NORMAL_TEMPLATES: tuple[str, ...] = (
        "Exame {kw}. Paciente {kw}.",
        "Laudo: {kw}. Nenhum achado relevante.",
        "O exame apresenta-se {kw}.",
        "Resultado dentro do esperado, {kw}.",
        "{kw}. Sem alterações significativas.",
    )

    _ATENCAO_TEMPLATES: tuple[str, ...] = (
        "Encontrado achado leve, {kw} seguimento.",
        "Quadro {kw}, recomenda-se {kw}.",
        "Observar com {kw} em consulta de retorno.",
        "Presença de alteração {kw}, indicado acompanhamento.",
        "{kw} do quadro, manter observação periódica.",
    )

    _URGENTE_TEMPLATES: tuple[str, ...] = (
        "{kw}! Necessária intervenção imediata.",
        "Sinais de {kw}, considerar internação.",
        "Quadro {kw} com risco à vida.",
        "Evidência de {kw}, encaminhar ao pronto-socorro.",
        "Achado {kw}, avaliar emergência clínica.",
    )

    def __init__(
        self,
        n_samples: int = 2000,
        random_state: int = 42,
        *,
        weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
    ) -> None:
        self.n_samples = n_samples
        self.random_state = random_state
        self.weights = weights

    @classmethod
    def _render(cls, template: str, keywords: tuple[str, ...], rng: random.Random) -> str:
        parts = template.split("{kw}")
        chosen = [rng.choice(keywords) for _ in range(len(parts) - 1)]
        return "".join(part for pair in zip(parts, chosen + [""], strict=True) for part in pair)

    def generate(self) -> pd.DataFrame:
        """Retorna um DataFrame com colunas `text` e `label`."""
        rng = random.Random(self.random_state)
        counts: dict[UrgencyLevel, int] = {
            UrgencyLevel.NORMAL: int(self.n_samples * self.weights[0]),
            UrgencyLevel.ATENCAO: int(self.n_samples * self.weights[1]),
            UrgencyLevel.URGENTE: self.n_samples
            - int(self.n_samples * self.weights[0])
            - int(self.n_samples * self.weights[1]),
        }

        templates_by_level: dict[UrgencyLevel, tuple[str, ...]] = {
            UrgencyLevel.NORMAL: self._NORMAL_TEMPLATES,
            UrgencyLevel.ATENCAO: self._ATENCAO_TEMPLATES,
            UrgencyLevel.URGENTE: self._URGENTE_TEMPLATES,
        }
        keywords_by_level: dict[UrgencyLevel, tuple[str, ...]] = {
            UrgencyLevel.NORMAL: KEYWORDS_NORMAL,
            UrgencyLevel.ATENCAO: KEYWORDS_ATENCAO,
            UrgencyLevel.URGENTE: KEYWORDS_URGENTE,
        }

        texts: list[str] = []
        labels: list[str] = []
        for level, count in counts.items():
            templates = templates_by_level[level]
            keywords = keywords_by_level[level]
            for _ in range(count):
                template = rng.choice(templates)
                texts.append(self._render(template, keywords, rng))
                labels.append(level.value)

        return pd.DataFrame({"text": texts, "label": labels})

    def save(self, path: str | Path) -> Path:
        """Gera e persiste o dataset sintético em CSV."""
        frame = self.generate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index=False)
        return target
