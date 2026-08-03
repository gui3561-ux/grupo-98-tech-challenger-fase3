from __future__ import annotations

from src.core.urgency import UrgencyLevel
from src.domain.interfaces import ModelInference
from src.infrastructure.predictors import ScikitLearnInference


def test_scikit_inference_predict_returns_valid_level(tmp_path) -> None:
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
    inference = ScikitLearnInference(pipeline)

    level, confidence = inference.predict(texts[1])
    assert isinstance(level, UrgencyLevel)
    assert level == UrgencyLevel.URGENTE
    assert 0.0 <= confidence <= 1.0
    assert isinstance(inference, ModelInference)
    assert inference.supports_batch() is True
