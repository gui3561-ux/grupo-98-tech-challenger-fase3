from __future__ import annotations

import json
import time
from pathlib import Path

from src.domain.interfaces import ModelInference
from src.infrastructure.predictors import ONNXInference, ScikitLearnInference


class BenchmarkResult:
    """Resultado agregado de um benchmark de latência."""

    def __init__(self, mean_ms: float, p95_ms: float, min_ms: float, max_ms: float) -> None:
        self.mean_ms = mean_ms
        self.p95_ms = p95_ms
        self.min_ms = min_ms
        self.max_ms = max_ms

    def to_dict(self) -> dict[str, float]:
        return {
            "mean_ms": round(self.mean_ms, 4),
            "p95_ms": round(self.p95_ms, 4),
            "min_ms": round(self.min_ms, 4),
            "max_ms": round(self.max_ms, 4),
        }


class InferenceBenchmarker:
    """Compara latência entre inferência nativa (Scikit) e otimizada (ONNX)."""

    def __init__(self, n_runs: int = 200, warmup: int = 20) -> None:
        self.n_runs = n_runs
        self.warmup = warmup

    def _measure(self, inference: ModelInference, texts: list[str]) -> BenchmarkResult:
        for _ in range(self.warmup):
            inference.predict(texts[0])
        latencies: list[float] = []
        for _ in range(self.n_runs):
            for text in texts:
                start = time.perf_counter()
                inference.predict(text)
                latencies.append((time.perf_counter() - start) * 1000.0)
        latencies.sort()
        p95_index = min(len(latencies) - 1, int(len(latencies) * 0.95))
        return BenchmarkResult(
            mean_ms=sum(latencies) / len(latencies),
            p95_ms=latencies[p95_index],
            min_ms=latencies[0],
            max_ms=latencies[-1],
        )

    def run(
        self,
        sklearn: ScikitLearnInference,
        onnx: ONNXInference,
        texts: list[str],
    ) -> dict[str, object]:
        """Executa o benchmark e retorna um relatório comparativo."""
        native = self._measure(sklearn, texts)
        optimized = self._measure(onnx, texts)
        speedup = native.mean_ms / optimized.mean_ms if optimized.mean_ms > 0 else float("inf")
        return {
            "native_sklearn": native.to_dict(),
            "optimized_onnx": optimized.to_dict(),
            "speedup_x": round(speedup, 4),
            "n_runs": self.n_runs,
        }

    def save(self, report: dict[str, object], path: str | Path) -> Path:
        """Persiste o relatório de benchmark em JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return target
