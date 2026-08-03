from __future__ import annotations

from src.core.urgency import UrgencyLevel
from src.services.data_generator import SyntheticDataGenerator


def test_generate_returns_expected_shape_and_labels() -> None:
    generator = SyntheticDataGenerator(n_samples=600, random_state=7)
    frame = generator.generate()

    assert frame.shape[0] == 600
    assert set(frame.columns) == {"text", "label"}
    assert set(frame["label"].unique()) == {
        UrgencyLevel.NORMAL.value,
        UrgencyLevel.ATENCAO.value,
        UrgencyLevel.URGENTE.value,
    }


def test_generate_is_reproducible_with_seed() -> None:
    first = SyntheticDataGenerator(n_samples=200, random_state=1).generate()
    second = SyntheticDataGenerator(n_samples=200, random_state=1).generate()
    assert first["text"].tolist() == second["text"].tolist()


def test_all_texts_are_non_blank() -> None:
    frame = SyntheticDataGenerator(n_samples=300, random_state=3).generate()
    assert all(len(text.strip()) > 0 for text in frame["text"])
