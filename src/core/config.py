from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class Settings:
    """Configurações globais do projeto, carregadas de variáveis de ambiente."""

    model_dir: Path = field(
        default_factory=lambda: Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models"))
    )
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
    )
    medical_abstracts_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("MEDICAL_ABSTRACTS_DIR", PROJECT_ROOT / "data" / "experiments" / "medical_abstracts")
        )
    )
    reports_dir: Path = field(
        default_factory=lambda: Path(os.getenv("REPORTS_DIR", PROJECT_ROOT / "reports"))
    )
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "urgency_classifier"))
    onnx_model_name: str = field(
        default_factory=lambda: os.getenv("ONNX_MODEL_NAME", "urgency_classifier.onnx")
    )
    n_samples: int = field(default_factory=lambda: int(os.getenv("N_SAMPLES", "2000")))
    random_state: int = field(default_factory=lambda: int(os.getenv("RANDOM_STATE", "42")))
    test_size: float = field(default_factory=lambda: float(os.getenv("TEST_SIZE", "0.2")))
    seed_text: str = field(
        default_factory=lambda: os.getenv(
            "SEED_TEXT", "Exame sem achados relevantes, tudo dentro da normalidade."
        )
    )

    def ensure_dirs(self) -> None:
        """Garante que os diretórios de saída existam."""
        for path in (self.model_dir, self.data_dir, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def sklearn_model_path(self) -> Path:
        return self.model_dir / f"{self.model_name}.joblib"

    @property
    def onnx_model_path(self) -> Path:
        return self.model_dir / self.onnx_model_name
