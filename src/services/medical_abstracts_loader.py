from __future__ import annotations

from pathlib import Path

import pandas as pd

# Mapeamento condition_label -> urgência, validado em
# notebooks/02_eda_medical_abstracts.ipynb (Estratégia 02): heurística documentada, não um
# protocolo de triagem clínica validado.
URGENCY_MAP: dict[int, str] = {
    1: "atencao",  # neoplasms
    2: "atencao",  # digestive system diseases
    3: "urgente",  # nervous system diseases
    4: "urgente",  # cardiovascular diseases
    5: "normal",  # general pathological conditions
}


class MedicalAbstractsLoader:
    """Carrega o Medical Abstracts TC Corpus e mapeia condition_label -> urgência.

    Replica a Estratégia 02 (ver notebooks/02_eda_medical_abstracts.ipynb e
    notebooks/EXPERIMENT_REPORT.md): concatena os splits oficiais, remove duplicatas exatas de
    texto (a fonte tem overlap treino/teste com rótulo divergente) e mapeia as 5 classes de
    especialidade clínica para as 3 classes de urgência exigidas pelo desafio.
    """

    def __init__(self, source_dir: str | Path) -> None:
        self.source_dir = Path(source_dir)

    def load(self) -> pd.DataFrame:
        """Retorna um DataFrame com colunas `text` e `label` (urgência), sem duplicatas."""
        train = pd.read_parquet(self.source_dir / "train.parquet")
        test = pd.read_parquet(self.source_dir / "test.parquet")
        combined = pd.concat([train, test], ignore_index=True)
        combined = combined.drop_duplicates(subset=["medical_abstract"]).reset_index(drop=True)
        combined["label"] = combined["condition_label"].map(URGENCY_MAP)
        return combined.rename(columns={"medical_abstract": "text"})[["text", "label"]]

    def save(self, path: str | Path) -> Path:
        """Carrega, mapeia e persiste o dataset em CSV com colunas `text`,`label`."""
        frame = self.load()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index=False)
        return target
