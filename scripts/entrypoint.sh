#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_DIR:-/app/models/urgency_classifier.onnx}"

if [ ! -f "$MODEL_PATH" ]; then
  echo "Modelo ONNX não encontrado. Executando pipeline de treino..."
  python -m src.cli.main generate-data
  python -m src.cli.main train
  python -m src.cli.main convert-onnx
  python -m src.cli.main benchmark
  echo "Pipeline concluído."
fi

exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000
