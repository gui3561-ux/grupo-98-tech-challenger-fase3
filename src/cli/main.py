from __future__ import annotations

import argparse
from pathlib import Path

from src.core.config import Settings
from src.infrastructure.predictors import ONNXInference, ScikitLearnInference
from src.services.benchmark import InferenceBenchmarker
from src.services.data_generator import SyntheticDataGenerator
from src.services.onnx_converter import ONNXConverter
from src.services.trainer import ModelTrainer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ml-pipeline", description="Pipeline de ML para triagem.")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate-data", help="Gera dataset sintético.")
    generate.add_argument("--output", type=Path, default=None)

    train = sub.add_parser("train", help="Treina o modelo Scikit-Learn.")
    train.add_argument("--data", type=Path, default=None)
    train.add_argument("--output", type=Path, default=None)

    convert = sub.add_parser("convert-onnx", help="Converte o modelo para ONNX.")
    convert.add_argument("--input", type=Path, default=None)
    convert.add_argument("--output", type=Path, default=None)

    benchmark = sub.add_parser("benchmark", help="Compara latência nativa vs ONNX.")
    benchmark.add_argument("--data", type=Path, default=None)
    benchmark.add_argument("--report", type=Path, default=None)

    return parser


def _resolve_data(settings: Settings) -> Path:
    candidates = [
        settings.data_dir / "laudos.csv",
        settings.data_dir / "medical_reports.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return settings.data_dir / "laudos.csv"


def _load_sample_texts(settings: Settings, data_path: Path) -> list[str]:
    import pandas as pd

    frame = pd.read_csv(data_path)
    return frame["text"].astype(str).head(50).tolist()


def cmd_generate_data(args: argparse.Namespace, settings: Settings) -> None:
    output = args.output or (settings.data_dir / "laudos.csv")
    generator = SyntheticDataGenerator(
        n_samples=settings.n_samples, random_state=settings.random_state
    )
    generator.save(output)
    print(f"Dataset gerado em {output}")


def cmd_train(args: argparse.Namespace, settings: Settings) -> None:
    data_path = args.data or _resolve_data(settings)
    output = args.output or settings.sklearn_model_path
    trainer = ModelTrainer(settings)
    texts, labels = trainer.load_data(data_path)
    pipeline, metrics = trainer.train(texts, labels)
    trainer.save(pipeline, output)
    print(f"Modelo salvo em {output}")
    print(f"Acurácia: {metrics.get('accuracy', 0):.4f}")


def cmd_convert(args: argparse.Namespace, settings: Settings) -> None:
    import joblib

    input_path = args.input or settings.sklearn_model_path
    output = args.output or settings.onnx_model_path
    raw_pipeline = joblib.load(input_path)
    converter = ONNXConverter()
    converter.convert(raw_pipeline, output)
    print(f"Modelo ONNX salvo em {output}")


def cmd_benchmark(args: argparse.Namespace, settings: Settings) -> None:
    data_path = args.data or _resolve_data(settings)
    report_path = args.report or (settings.reports_dir / "benchmark_report.json")
    texts = _load_sample_texts(settings, data_path)
    sklearn = ScikitLearnInference.from_path(settings.sklearn_model_path)
    onnx = ONNXInference(settings.onnx_model_path)
    benchmarker = InferenceBenchmarker()
    report = benchmarker.run(sklearn, onnx, texts)
    benchmarker.save(report, report_path)
    print(f"Relatório salvo em {report_path}")
    print(f"Speedup: {report['speedup_x']}x")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    settings = Settings()
    settings.ensure_dirs()

    handlers = {
        "generate-data": cmd_generate_data,
        "train": cmd_train,
        "convert-onnx": cmd_convert,
        "benchmark": cmd_benchmark,
    }
    handlers[args.command](args, settings)


if __name__ == "__main__":
    main()
