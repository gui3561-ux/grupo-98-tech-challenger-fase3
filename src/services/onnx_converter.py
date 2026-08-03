from __future__ import annotations

from pathlib import Path
from typing import Any

import onnx
from onnx import ModelProto
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType


class ONNXConverter:
    """Converte um pipeline Scikit-Learn em um modelo ONNX serializável."""

    @staticmethod
    def convert(pipeline: Any, output_path: str | Path) -> Path:
        """Converte o pipeline em ONNX e salva no caminho indicado."""
        initial_types: list[tuple[str, StringTensorType]] = [
            ("input", StringTensorType([None, 1]))
        ]
        options: dict[int, dict[str, bool]] = {id(pipeline): {"zipmap": False}}
        model: ModelProto = convert_sklearn(
            pipeline,
            initial_types=initial_types,
            target_opset=17,
            options=options,
        )
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        onnx.save_model(model, target)
        return target
