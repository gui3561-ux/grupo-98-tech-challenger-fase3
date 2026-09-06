from __future__ import annotations

import pandas as pd

from src.services.medical_abstracts_loader import MedicalAbstractsLoader


def _write_parquet_fixtures(source_dir) -> None:
    train = pd.DataFrame(
        {
            "medical_abstract": ["texto A", "texto B", "texto C"],
            "condition_label": [1, 3, 5],
        }
    )
    test = pd.DataFrame(
        {
            "medical_abstract": ["texto A", "texto D"],  # "texto A" duplica o treino
            "condition_label": [1, 4],
        }
    )
    train.to_parquet(source_dir / "train.parquet")
    test.to_parquet(source_dir / "test.parquet")


def test_load_maps_condition_to_urgency_and_dedupes(tmp_path) -> None:
    _write_parquet_fixtures(tmp_path)
    loader = MedicalAbstractsLoader(tmp_path)

    frame = loader.load()

    assert set(frame.columns) == {"text", "label"}
    assert len(frame) == 4  # 5 linhas originais - 1 duplicata exata de texto
    assert frame[frame["text"] == "texto A"]["label"].iloc[0] == "atencao"  # condition_label=1
    assert frame[frame["text"] == "texto B"]["label"].iloc[0] == "urgente"  # condition_label=3
    assert frame[frame["text"] == "texto C"]["label"].iloc[0] == "normal"  # condition_label=5
    assert frame[frame["text"] == "texto D"]["label"].iloc[0] == "urgente"  # condition_label=4
    assert set(frame["label"].unique()) <= {"normal", "atencao", "urgente"}


def test_save_persists_csv_with_expected_columns(tmp_path) -> None:
    _write_parquet_fixtures(tmp_path)
    loader = MedicalAbstractsLoader(tmp_path)
    output = tmp_path / "output" / "medical_reports.csv"

    loader.save(output)

    assert output.exists()
    frame = pd.read_csv(output)
    assert list(frame.columns) == ["text", "label"]
    assert len(frame) == 4
