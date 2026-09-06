from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.core.config import Settings


class ModelTrainer:
    """Treina o pipeline TF-IDF + SelectKBest + LogisticRegression e o persiste.

    Configuração validada em notebooks/06_model_comparison.ipynb e
    notebooks/07_final_benchmark.ipynb: vence o baseline RandomForest em macro F1 e é ~213x
    menor / ~39x mais rápido nativamente; `SelectKBest(chi2, k=8000)` depois do TF-IDF de 20000
    features melhora ainda mais o macro F1 e reduz tamanho/latência do modelo convertido para ONNX.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

    def load_data(self, csv_path: str | Path) -> tuple[list[str], np.ndarray]:
        """Carrega textos e rótulos de um CSV com colunas `text` e `label`."""
        import pandas as pd

        frame = pd.read_csv(csv_path)
        return frame["text"].astype(str).tolist(), frame["label"].to_numpy()

    def _build_pipeline(self) -> Pipeline[list[str], np.ndarray]:
        return Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")),
                ("select", SelectKBest(chi2, k=8000)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=self.settings.random_state,
                    ),
                ),
            ]
        )

    def train(
        self,
        texts: list[str],
        labels: np.ndarray,
    ) -> tuple[Pipeline[list[str], np.ndarray], dict[str, float]]:
        """Treina o pipeline e retorna (modelo, métricas de avaliação)."""
        x_train, x_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=self.settings.test_size,
            random_state=self.settings.random_state,
            stratify=labels,
        )
        pipeline = self._build_pipeline()
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        metrics: dict[str, float] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
        }
        self._attach_report(metrics, y_test, y_pred)
        return pipeline, metrics

    def _attach_report(self, metrics: dict[str, float], y_test: np.ndarray, y_pred: np.ndarray) -> None:
        report: dict[str, dict[str, float]] = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        for label, values in report.items():
            if isinstance(values, dict) and "f1-score" in values:
                metrics[f"f1_{label}"] = float(values["f1-score"])

    def save(self, pipeline: Pipeline[list[str], np.ndarray], path: str | Path) -> Path:
        """Serializa o pipeline treinado com joblib."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, target)
        return target
