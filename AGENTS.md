# AGENTS.md

NLP + ONNX + FastAPI system that classifies medical reports into urgency levels
(`normal` / `atencao` / `urgente`). Layered, SOLID architecture. See `README.md`
for the full write-up and architecture table.

## Commands (run from repo root)

Dependency management uses [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`); there is no `requirements.txt` anymore.

```bash
uv sync --group dev   # creates .venv, installs runtime + dev deps (pytest/httpx/ruff)
cp .env.example .env

uv run ruff check .          # lint (CI)
uv run pytest --tb=short     # tests (CI); testpaths=tests, quiet by default
```

ML pipeline CLI — order matters (`generate-data` → `train` → `convert-onnx` → `benchmark`):

```bash
uv run python -m src.cli.main generate-data   # loads+maps Medical Abstracts TC Corpus -> data/medical_reports.csv
uv run python -m src.cli.main train           # TF-IDF(20k) -> SelectKBest(chi2, k=8000) -> LogisticRegression -> models/*.joblib
uv run python -m src.cli.main convert-onnx    # -> models/urgency_classifier.onnx
uv run python -m src.cli.main benchmark       # -> reports/benchmark_report.json
```

`generate-data` reads `data/experiments/medical_abstracts/{train,test,labels}.parquet`
(`MedicalAbstractsLoader`, `src/services/medical_abstracts_loader.py`) and maps
`condition_label` -> urgency per the mapping validated in `notebooks/02_eda_medical_abstracts.ipynb`.
The legacy `SyntheticDataGenerator` (`src/services/data_generator.py`, writes `data/laudos.csv`)
still exists and is still tested, but is no longer wired into the CLI or the Airflow DAG.

API dev server: `uv run uvicorn src.api.main:app --reload` (needs `models/urgency_classifier.onnx` to exist or `/predict` fails).

## Key facts an agent would miss

- **Config is env-driven, not code-driven.** `Settings` (`src/core/config.py`) is a frozen
  dataclass reading `MODEL_DIR`, `DATA_DIR`, `MEDICAL_ABSTRACTS_DIR`, `REPORTS_DIR`, `MODEL_NAME`,
  `ONNX_MODEL_NAME`, `N_SAMPLES`, `RANDOM_STATE`, `TEST_SIZE`, `SEED_TEXT` from the environment
  (via `.env`). `N_SAMPLES` only affects the legacy synthetic generator, not the default pipeline.
  `PROJECT_ROOT` is derived from the file path (3 parents up), so paths resolve to the repo root
  regardless of CWD. Tests instantiate `Settings` directly with `tmp_path` dirs to isolate artifacts.
- **ONNX class order is NOT the enum declaration order.** `skl2onnx` orders probability columns
  alphabetically by `pipeline.classes_` (`atencao, normal, urgente`), not by `UrgencyLevel`'s
  declaration order (`normal, atencao, urgente`). `ONNXInference` (`src/infrastructure/predictors.py`)
  reads the real order from the ONNX graph's `classlabels_strings` attribute — never hardcode or
  assume an order here again; it silently swapped `normal`/`atencao` predictions before this fix
  (regression test: `tests/test_inference.py::test_onnx_inference_matches_native_class_order`).
- **The final Docker image needs the `en_US.UTF-8` locale.** `onnxruntime`'s `StringNormalizer` op
  (from the TF-IDF's lowercasing, via `skl2onnx`) requires it at session-init time; without it the
  container crashes on boot. Don't remove the `locales`/`locale-gen` layer from the `Dockerfile`.
- **The API requires a pre-trained ONNX model.** `docker-compose` runs `scripts/entrypoint.sh`
  which auto-runs the full pipeline only if the ONNX file is absent. In local dev you must run the
  pipeline yourself first. Test `tests/test_api.py` builds a tiny model via `ONNXConverter` in a fixture.
- **The CLI must be invoked as `python -m src.cli.main`** (package-relative imports); a bare
  `src/cli/main.py` will not work.
- **Tests need `onnxruntime`** to convert the tiny model; tests use a `DecisionTreeClassifier`,
  not the real model. Test artifacts go to `tmp_path`.
- **Airflow DAG must be copied into `$AIRFLOW_HOME/dags`.** Local setup:
  `export AIRFLOW_HOME="$PWD/airflow"` then `airflow db init`, create user, and copy
  `airflow/dags/retrain_dag.py` into the dags dir. The repo `airflow/` folder is the home, not a source tree.
- **`.env` is gitignored.** Copy `.env.example`; don't commit secrets.

## Conventions

- Python 3.11, type hints on every function/method (repo rule).
- Ruff config in `pyproject.toml`: line-length 100, ignores `E501`/`B008`; `tests/*` ignores `S101`.
  Follow existing style — don't reformat broadly.
- Layers: `core` (config/enums) → `domain` (ABC/Protocol/DTOs) → `infrastructure` (predictors)
  → `services` (pipeline steps) → `api` (routes/DI) → `cli`. New inference strategies implement
  `ModelInference` (`src/domain/interfaces.py`); don't couple routes to concrete predictors.
- CI (`.github/workflows/ci.yml`): `ruff check .` then `pytest --tb=short` on push/PR to `main`.
