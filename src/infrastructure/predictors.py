from __future__ import annotations

from typing import Any

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from src.core.urgency import UrgencyLevel
from src.domain.interfaces import ModelInference


class ScikitLearnInference(ModelInference):
    """Preditor baseado em um pipeline Scikit-Learn serializado com joblib."""

    def __init__(self, model: Pipeline[Any, Any]) -> None:
        self._model = model
        classes: list[Any] = list(model.classes_)
        self._label_to_level: dict[Any, UrgencyLevel] = {
            str(label): UrgencyLevel(label) for label in classes
        }

    @classmethod
    def from_path(cls, path: str | Any) -> ScikitLearnInference:
        """Carrega um pipeline serializado e retorna o preditor."""
        model = joblib.load(path)
        return cls(model)

    def _to_level(self, label: str) -> UrgencyLevel:
        return self._label_to_level[label]

    def predict(self, text: str) -> tuple[UrgencyLevel, float]:
        proba = self.predict_proba(text)
        level = max(proba, key=proba.get)  # type: ignore[arg-type]
        return level, proba[level]

    def predict_proba(self, text: str) -> dict[UrgencyLevel, float]:
        probs: list[float] = self._model.predict_proba([text])[0]
        return {self._to_level(str(label)): float(p) for label, p in zip(self._model.classes_, probs, strict=True)}

    def supports_batch(self) -> bool:
        return True


class ONNXInference(ModelInference):
    """Preditor baseado em um modelo ONNX executado via onnxruntime."""

    def __init__(self, onnx_path: str | Any) -> None:
        import onnx
        import onnxruntime as ort

        self._sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self._input_name = self._sess.get_inputs()[0].name
        self._output_name = self._select_probability_output()
        self._class_order = self._read_class_order(onnx.load(str(onnx_path)))

    def _select_probability_output(self) -> str:
        """Seleciona a saída de probabilidades (tensor float) do modelo ONNX."""
        for output in self._sess.get_outputs():
            if output.type.startswith("tensor(float"):
                return output.name
        raise RuntimeError("Nenhuma saída de probabilidades encontrada no modelo ONNX.")

    @staticmethod
    def _read_class_order(model: Any) -> tuple[UrgencyLevel, ...]:
        """Lê a ordem real das classes no grafo ONNX (`classlabels_strings` do nó classificador).

        Não assume ordem alfabética nem a ordem de declaração do enum: o skl2onnx ordena as
        colunas de probabilidade conforme `pipeline.classes_` (alfabética para rótulos string),
        que não bate com a ordem de declaração de `UrgencyLevel` — usar essa suposição trocava
        as probabilidades de `normal` e `atencao` silenciosamente (bug corrigido aqui).
        """
        for node in model.graph.node:
            for attr in node.attribute:
                if attr.name == "classlabels_strings":
                    return tuple(UrgencyLevel(label.decode("utf-8")) for label in attr.strings)
        raise RuntimeError("Atributo classlabels_strings não encontrado no grafo ONNX.")

    def predict(self, text: str) -> tuple[UrgencyLevel, float]:
        proba = self.predict_proba(text)
        level = max(proba, key=proba.get)  # type: ignore[arg-type]
        return level, proba[level]

    def predict_proba(self, text: str) -> dict[UrgencyLevel, float]:
        inputs = {self._input_name: np.array([text], dtype=np.str_)[:, None]}
        probs: np.ndarray = self._sess.run([self._output_name], inputs)[0]
        scores = probs.ravel().tolist()
        return {label: float(scores[i]) for i, label in enumerate(self._class_order)}

    def supports_batch(self) -> bool:
        return True
