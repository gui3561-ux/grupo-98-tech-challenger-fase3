from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.core.config import Settings
from src.services.medical_abstracts_loader import MedicalAbstractsLoader
from src.services.onnx_converter import ONNXConverter
from src.services.trainer import ModelTrainer

DEFAULT_ARGS: dict[str, Any] = {
    "owner": "fiap-mle",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}

dag = DAG(
    dag_id="retrain_dag",
    default_args=DEFAULT_ARGS,
    description="Ingestão, treino e exportação ONNX do classificador de laudos.",
    schedule="@daily",
    catchup=False,
    tags=["ml", "retrain"],
)


def _ingest_data(**context: Any) -> str:
    settings: Settings = context["settings"]
    loader = MedicalAbstractsLoader(settings.medical_abstracts_dir)
    output = settings.data_dir / "medical_reports.csv"
    loader.save(output)
    return str(output)


def _train_model(data_path: str, **context: Any) -> str:
    settings: Settings = context["settings"]
    trainer = ModelTrainer(settings)
    texts, labels = trainer.load_data(data_path)
    pipeline, _metrics = trainer.train(texts, labels)
    output = settings.sklearn_model_path
    trainer.save(pipeline, output)
    return str(output)


def _convert_onnx(model_path: str, **context: Any) -> str:
    import joblib

    settings: Settings = context["settings"]
    raw_pipeline = joblib.load(model_path)
    output = settings.onnx_model_path
    ONNXConverter.convert(raw_pipeline, output)
    return str(output)


ingest_task = PythonOperator(
    task_id="ingest_data",
    python_callable=_ingest_data,
    op_kwargs={"settings": Settings()},
    dag=dag,
)

train_task = PythonOperator(
    task_id="train_model",
    python_callable=_train_model,
    op_kwargs={
        "data_path": "{{ ti.xcom_pull(task_ids='ingest_data') }}",
        "settings": Settings(),
    },
    dag=dag,
)

convert_task = PythonOperator(
    task_id="convert_onnx",
    python_callable=_convert_onnx,
    op_kwargs={
        "model_path": "{{ ti.xcom_pull(task_ids='train_model') }}",
        "settings": Settings(),
    },
    dag=dag,
)

ingest_task >> train_task >> convert_task
