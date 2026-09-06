from __future__ import annotations

import pytest

from src.core.urgency import UrgencyLevel
from src.domain.interfaces import ModelInference
from src.infrastructure.predictors import ONNXInference, ScikitLearnInference


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


def test_onnx_inference_matches_native_class_order(tmp_path) -> None:
    """Regressão: o ONNX ordena as classes alfabeticamente (atencao, normal, urgente), não na
    ordem de declaração de `UrgencyLevel` (normal, atencao, urgente) — usar a ordem errada troca
    silenciosamente as probabilidades de `normal` e `atencao`."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from src.services.onnx_converter import ONNXConverter

    texts = [
        "exame normal sem achados relevantes",
        "achado grave emergência internação imediata",
        "quadro leve recomenda acompanhamento periódico",
        "tudo dentro da normalidade, sem alterações",
        "sinais de risco à vida, encaminhar emergência",
        "alteração moderada, manter observação",
    ]
    labels = ["normal", "urgente", "atencao", "normal", "urgente", "atencao"]
    pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
    pipeline.fit(texts, labels)

    onnx_path = tmp_path / "model.onnx"
    ONNXConverter.convert(pipeline, onnx_path)
    onnx_inference = ONNXInference(onnx_path)

    for text in texts:
        native_level = UrgencyLevel(pipeline.predict([text])[0])
        onnx_level, _confidence = onnx_inference.predict(text)
        assert onnx_level == native_level, f"ONNX e nativo divergem para: {text!r}"

        # Tolerância larga: ONNX Runtime calcula em float32 e o modelo é minúsculo (poucas
        # amostras), então pequenas diferenças numéricas são esperadas. O que a regressão
        # protege é a ORDEM das classes, não a paridade numérica exata.
        native_proba = dict(
            zip(pipeline.classes_, pipeline.predict_proba([text])[0], strict=True)
        )
        onnx_proba = onnx_inference.predict_proba(text)
        for level in UrgencyLevel:
            assert onnx_proba[level] == pytest.approx(native_proba[level.value], abs=0.2)
