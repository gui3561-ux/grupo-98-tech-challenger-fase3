# 🏥 Sistema de Triagem de Laudos Médicos (NLP + ONNX + FastAPI + Observabilidade)

**Tech Challenge — Fase 3 · Pós-Graduação em Machine Learning Engineering · FIAP**
**Grupo 98**

Sistema automatizado de triagem de laudos médicos em três categorias de urgência
(`normal`, `atencao`, `urgente`), construído com princípios SOLID, Clean Code e
arquitetura em camadas. Utiliza NLP leve (TF-IDF + Random Forest), otimização de
inferência com **ONNX Runtime**, serviço via **FastAPI**, observabilidade com
**Prometheus/Grafana**, orquestração com **Airflow** e CI/CD via **GitHub Actions**.

---

## 📑 Sumário

- [Arquitetura](#-arquitetura-e-separação-de-responsabilidades)
- [Decisão Arquitetural em Nuvem](#-decisão-arquitetural-em-nuvem-batch-vs-real-time)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Configuração Rápida](#-configuração-rápida)
- [Pipeline de ML (gerar → treinar → ONNX → benchmark)](#-pipeline-de-ml)
- [API FastAPI e Métricas](#-api-fastapi-e-métricas)
- [Docker Compose (API + Prometheus + Grafana)](#-docker-compose)
- [Airflow DAG](#-airflow-dag)
- [CI/CD (GitHub Actions)](#-cicd-github-actions)
- [Resultados de Latência (Nativo vs ONNX)](#-resultados-de-latência)
- [Roteiro STAR para o Vídeo (5 min)](#-roteiro-star-para-o-vídeo-de-5-minutos)

---

## 🧩 Arquitetura e Separação de Responsabilidades

O projeto segue uma arquitetura em camadas orientada a domínio, com aplicação
rigorosa dos princípios SOLID:

| Camada | Responsabilidade | Pasta |
|--------|------------------|-------|
| **core** | Enums de urgência, configurações globais (pydantic/dataclass) | `src/core/` |
| **domain** | Contratos Pydantic (DTOs), interfaces abstratas (ABC) e Protocol | `src/domain/` |
| **infrastructure** | Implementações concretas de predição (Scikit e ONNX) | `src/infrastructure/` |
| **services** | Geração de dados, treino, conversão ONNX e benchmarking | `src/services/` |
| **api** | Rotas FastAPI, injeção de dependências e exportação de métricas | `src/api/` |
| **cli** | Entrypoints de linha de comando do pipeline de ML | `src/cli/` |

### Princípios SOLID aplicados

1. **Single Responsibility** — cada módulo tem uma única razão para mudar:
   *data generation*, *training*, *inference execution*, *API routes* e
   *metrics export* estão isolados.
2. **Open/Closed** — novas estratégias de inferência (ex.: TensorRT, XGBoost)
   são adicionadas implementando a interface `ModelInference`, sem alterar a API.
3. **Interface Segregation** — `ModelInference` (ABC) e `MetricsExporter`
   (Protocol) expõem apenas os métodos necessários.
4. **Dependency Inversion** — a API FastAPI injeta as dependências
   (`get_inference`, `get_metrics`, `get_settings`) via `Depends` + `Annotated`,
   sem acoplar as rotas a implementações concretas.

### Exemplo da interface de inferência (`src/domain/interfaces.py`)

```python
class ModelInference(ABC):
    @abstractmethod
    def predict(self, text: str) -> tuple[UrgencyLevel, float]: ...
    @abstractmethod
    def predict_proba(self, text: str) -> dict[UrgencyLevel, float]: ...
    @abstractmethod
    def supports_batch(self) -> bool: ...
```

Implementações: `ScikitLearnInference` e `ONNXInference` (em `src/infrastructure/`).

---

## ☁️ Decisão Arquitetural em Nuvem (Batch vs. Real-Time)

A escolha entre **Batch** e **Real-Time** para o deploy de um sistema de triagem
médica depende dos requisitos não funcionais dominantes: **latência clínica**,
**volume de laudos** e **SLA de disponibilidade**.

### Comparativo (AWS / Azure / GCP)

| Critério | **Batch** | **Real-Time (Synchronous)** |
|----------|-----------|------------------------------|
| **Latência** | Minutos a horas (janela agendada) | Milissegundos a segundos (síncrono) |
| **Custo** | Menor (instâncias spot/lote) | Maior (instâncias dedicadas, autoscaling) |
| **Complexidade** | Baixa (scheduler + storage) | Média-alta (API, load balancer, caching) |
| **AWS** | AWS Batch + S3 + EventBridge | SageMaker Endpoint / Lambda + API Gateway |
| **Azure** | Azure ML Batch Endpoints | Azure ML Online Endpoint + APIM |
| **GCP** | Vertex AI Batch Prediction | Vertex AI Endpoint + Cloud Run |
| **Uso médico** | Relatórios diários, auditoria, triagem retrospectiva | Triagem em tempo real no pronto-socorro |

### Recomendação para o contexto hospitalar

Para a **triagem em tempo real de laudos no atendimento**, a estratégia mais
adequada é **Real-Time síncrono** — o médico ou o sistema de prontuário
(eletrônico) submete o laudo e recebe a categoria de urgência imediatamente.

**Recomendação (AWS como referência):**
- **SageMaker Endpoint** (ou ECS/Fargate com contêiner ONNX) para inferência em
  tempo real, com **autoscaling** por latência.
- **API Gateway** para autenticação, rate-limiting e exposição segura do endpoint.
- **CloudWatch / Prometheus** para observabilidade.
- **Complemento Batch:** re-treino e re-avaliação periódica via **AWS Batch** /
  **SageMaker Processing**, acionado pelo **Airflow (MWAA)**, com ingestão do
  S3 e publicação do modelo atualizado.

> **Conclusão:** adote **Real-Time** para o serviço de predição ativo (o que este
> repositório implementa) e **Batch** para o ciclo de treino/atualização do modelo,
> separando claramente os dois caminhos.

---

## 📂 Estrutura do Repositório

```
.
├── .github/workflows/ci.yml          # CI: lint (ruff) → teste (pytest)
├── airflow/dags/retrain_dag.py       # DAG de retreino (ingestão→treino→ONNX)
├── src/
│   ├── core/                         # config.py, urgency.py (StrEnum)
│   ├── domain/                       # interfaces.py, schemas.py, metrics_protocol.py
│   ├── services/                     # data_generator, trainer, onnx_converter, benchmark
│   ├── infrastructure/               # predictors.py (Scikit + ONNX)
│   ├── api/                          # main, routes, dependencies, metrics
│   └── cli/                          # main.py (entrypoints do pipeline)
├── tests/                            # pytest (API, schemas, inferência, dados)
├── metrics/
│   ├── prometheus.yml                # scrape config
│   ├── dashboards/urgency_dashboard.json
│   └── grafana/                      # provisionamento (datasource + provider)
├── docker-compose.yml                # api + prometheus + grafana
├── Dockerfile                        # multi-stage otimizado
├── pyproject.toml                    # deps (uv) + ruff + pytest
├── uv.lock
├── .env.example
└── README.md
```

---

## ⚙️ Configuração Rápida

### Opção A — Rodar com Docker Compose (recomendado)

```bash
docker compose up --build
```

O entrypoint detecta a ausência do modelo ONNX e executa o pipeline completo
(genera → train → convert → benchmark) antes de subir a API.

| Serviço | URL |
|---------|-----|
| **API (Swagger UI)** | http://localhost:8000/docs |
| **Healthcheck** | http://localhost:8000/health |
| **Métricas Prometheus** | http://localhost:8000/metrics |
| **Prometheus** | http://localhost:9090 |
| **Grafana** | http://localhost:3000 (admin/admin) |

### Opção B — Ambiente virtual (desenvolvimento)

Requer o [uv](https://docs.astral.sh/uv/) instalado.

```bash
uv sync --group dev
cp .env.example .env

# Executar o pipeline de ML
uv run python -m src.cli.main generate-data
uv run python -m src.cli.main train
uv run python -m src.cli.main convert-onnx
uv run python -m src.cli.main benchmark

# Subir a API
uv run uvicorn src.api.main:app --reload
```

---

## 🤖 Pipeline de ML

O pipeline é exposto pelo módulo CLI (`python -m src.cli.main`):

| Comando | Descrição |
|---------|-----------|
| `generate-data` | Gera **2.000** laudos sintéticos (50% normal, 30% atenção, 20% urgente) → `data/laudos.csv` |
| `train` | Treina pipeline **TF-IDF + Random Forest** e salva `.joblib` |
| `convert-onnx` | Converte o pipeline em **ONNX** (opset 17, saída de probabilidades) |
| `benchmark` | Compara latência **Scikit vs ONNX** e gera relatório JSON |

> Caso um CSV externo seja fornecido, basta apontá-lo em `train --data` (colunas
> `text` e `label`).

---

## 🔌 API FastAPI e Métricas

### Endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| `POST` | `/predict` | Recebe `{"text": "..."}`, executa inferência ONNX e retorna nível de urgência + latência |
| `GET` | `/health` | Healthcheck |
| `GET` | `/metrics` | Métricas no formato Prometheus |

### Exemplo de requisição

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Achado grave, com evidência de sangramento. Encaminhar ao pronto-socorro."}'
```

```json
{
  "level": "urgente",
  "label": "urgente",
  "confidence": 0.94,
  "latency_ms": 0.75
}
```

### Métricas expostas

- `urgency_http_requests_total{status}` — contagem de requisições por status HTTP (**Counter**)
- `urgency_request_latency_ms` — latência das requisições (**Histogram**)
- `urgency_predictions_total{level}` — predições por categoria de urgência (**Counter**)

As métricas de **requisição e latência** são registradas por um **middleware HTTP**
(instrumentação transparente de todos os endpoints), enquanto a contagem de
predições é registrada no próprio handler `/predict`. Isso segue a prática
recomendada pelo cliente Prometheus e elimina a contagem manual por rota.

### DTOs (Pydantic v2) em `src/domain/schemas.py`

- `PredictionRequest` — valida `text` (não vazio, limite de 5.000 chars, strip de espaços).
- `PredictionResponse` — `level`, `label`, `confidence`, `latency_ms`.

---

## 🐳 Docker Compose

`docker-compose.yml` orquestra três serviços na mesma rede `urgency-net`:

- **api** — portas `8000`
- **prometheus** — porta `9090` (scrape do `/metrics` da API a cada 5s)
- **grafana** — porta `3000` (credenciais `admin`/`admin`)

O **Grafana** provisiona automaticamente o datasource `Prometheus` e o dashboard
**"Triagem de Laudos Médicos"** (arquivo `metrics/dashboards/urgency_dashboard.json`)
com 3 painéis:

1. **Total de Requisições por Status** — `sum(rate(urgency_http_requests_total[1m])) by (status)`
2. **Latência da API** — p95 e média (via histograma)
3. **Predições por Nível de Urgência** — `sum(rate(urgency_predictions_total[5m])) by (level)`

---

## ⏰ Airflow DAG

A DAG `retrain_dag` (`airflow/dags/retrain_dag.py`) executa diariamente:

1. `ingest_data` → gera/atualiza o dataset sintético
2. `train_model` → treina o pipeline base
3. `convert_onnx` → converte para ONNX e salva

As tarefas trocam artefatos via **XCom** e seguem a ordem `ingest >> train >> convert`.

### Como acionar (local)

```bash
# Inicializar o Airflow (apenas a primeira vez)
export AIRFLOW_HOME="$PWD/airflow"
airflow db init
airflow users create \
  --username admin --password admin --firstname Admin \
  --lastname Admin --role Admin --email admin@example.com

# Copiar a DAG e iniciar o scheduler + webserver
cp airflow/dags/retrain_dag.py "$AIRFLOW_HOME/dags/"
airflow scheduler & 
airflow webserver --port 8080
```

Acesse `http://localhost:8080`, ative a DAG `retrain_dag` e use **Trigger DAG** para
executá-la manualmente.

---

## 🚀 CI/CD (GitHub Actions)

O workflow `.github/workflows/ci.yml` roda em `push` e `pull_request` para `main`:

1. **Lint** — `ruff check .`
2. **Testes** — `pytest`

Ambiente: `ubuntu-latest`, Python `3.11`, com cache de dependências pip.

---

## 📊 Resultados de Latência

Benchmark executado no pipeline real (2.000 amostras, 200 runs + 20 warmup):

| Métrica | Nativo (Scikit) | Otimizado (ONNX) | Speedup |
|---------|-----------------|------------------|---------|
| **Média** | 14.31 ms | 0.0143 ms | **~1000×** |
| **p95** | 14.75 ms | 0.0172 ms | ~857× |
| **Mín** | 11.54 ms | 0.0118 ms | — |
| **Máx** | 183.03 ms | 0.3313 ms | ~553× |

> **Interpretação:** para **inferência de amostra única** (caso de uso em tempo
> real), o ONNX Runtime entrega uma redução de latência de **cerca de 3 ordens de
> grandeza**, mantendo a mesma acurácia, pois o modelo serializado é idêntico.
> Isso viabiliza a triagem síncrona dentro do SLA de um pronto-socorro.

O relatório completo fica em `reports/benchmark_report.json` após a execução.

---

## 🎬 Roteiro STAR para o Vídeo de 5 Minutos

### **S**ituation (Contexto)
> Hospitais e pronto-socorros precisam priorizar laudos médicos rapidamente, mas
> a triagem manual é lenta, sujeita a erro humano e não escala com o volume de
> exames. O problema: classificar automaticamente cada laudo em **Normal**,
> **Atenção** ou **Urgente** para direcionar o atendimento.

### **T**ask (Tarefa)
> Construir um sistema de ML que classifique laudos em três níveis de urgência,
> servido como API, otimizado para baixa latência, observável e orquestrado —
> seguindo boas práticas de engenharia de software (SOLID), MLOps e CI/CD.

### **A**ction (Ação)
> 1. **Dados:** gerei 2.000 laudos sintéticos rotulados (normal/atenção/urgente).
> 2. **Modelo:** pipeline TF-IDF + Random Forest (Scikit-Learn).
> 3. **Otimização:** converti para ONNX Runtime, alcançando ~1000× de redução de
>    latência em amostra única.
> 4. **API:** FastAPI com injeção de dependências e DTOs Pydantic v2.
> 5. **Observabilidade:** Prometheus + Grafana (requisições por status, latência
>    p95, predições por urgência).
> 6. **Orquestração:** DAG do Airflow para retreino automático.
> 7. **CI/CD:** GitHub Actions (ruff + pytest) a cada push/PR.

### **R**esult (Resultado)
> API de triagem com inferência em **~0.01 ms** (ONNX vs ~14 ms nativo), endpoints
> documentados em Swagger, dashboards de monitoramento provisionados e pipeline
> de retreino automatizado — pronto para deploy em tempo real na nuvem.

---

## 🧪 Qualidade e Ferramentas

- **Lint/Format:** `ruff check .`
- **Testes:** `pytest`
- **Type hints:** em 100% das funções/métodos.
- **Validação:** Pydantic v2.

```bash
ruff check .
pytest --tb=short
```

---

## 📄 Licença

Projeto acadêmico (FIAP). Uso exclusivo para fins educacionais.
