# Notebooks — EDA e experimentação (branch `experiment/eda-model-selection`)

Área de **exploração e experimentação**. Nada aqui é código de produção; nada nesta branch altera `src/`,
`Dockerfile`, `docker-compose.yml`, `airflow/` ou `.github/workflows/`. Ver `EXPERIMENT_REPORT.md` na raiz do
repositório para o relatório consolidado.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-experiments.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/<nome>.ipynb   # para reexecutar um notebook
```

## Ordem e dependências entre notebooks

| Notebook | Depende de | Produz |
|---|---|---|
| `01_eda_laudos_sinteticos.ipynb` | `data/laudos.csv` (já no repo) | `data/experiments/results/eda_laudos_summary.json` |
| `02_eda_medical_abstracts.ipynb` | `data/experiments/medical_abstracts/*.parquet` (ver `SOURCE.md` na mesma pasta) | `data/experiments/results/eda_medical_abstracts_summary.json` |
| `03_baseline_models.ipynb` | `data/laudos.csv`, `data/experiments/medical_abstracts/*.parquet` | `data/experiments/results/baseline_results.json` |
| `04_dataset_comparison.ipynb` | os 3 JSONs acima | `data/experiments/results/dataset_recommendation.json` |
| `05_model_comparison.ipynb` *(fase B)* | resultados de 01-04 | — |
| `06_final_benchmark.ipynb` *(fase B)* | resultados de 05 | — |

Cada notebook lê os resumos numéricos dos anteriores via JSON em `data/experiments/results/` — nenhuma
conclusão depende de números digitados à mão em célula de markdown.

## Sobre o MIMIC-III (e o SemClinBr)

O pedido original comparava MIMIC-III vs. Medical Abstracts TC Corpus. Antes de escrever qualquer notebook,
verificamos na fonte oficial (PhysioNet) que a única versão do MIMIC-III sem credenciamento (o demo de 100
pacientes) não inclui a tabela `NOTEEVENTS` (texto livre) — confirmado diretamente na documentação do dataset e
validado baixando o arquivo (contém só o cabeçalho, 0 linhas). Um dataset derivado do Kaggle (`structured
medical records`, também investigado) não resolve isso: as "notas" nele são texto gerado a partir das tabelas
estruturadas do próprio demo, cobrindo só 9 internações reais (de 129 no demo), 100% de admissão tipo
`EMERGENCY` — sem variação de classe, inutilizável como target. O SemClinBr (corpus clínico real em português)
também foi avaliado: o corpus anotado exige formulário de requisição e aprovação da equipe da PUCPR, mesmo
padrão de credenciamento do MIMIC-III. Nenhum dos dois está disponível nesta branch hoje.

O MIMIC-III completo exige credenciamento PhysioNet (conta + treinamento CITI + assinatura do Data Use
Agreement), fora do escopo de uma exploração automatizada. Combinado com o time, a comparação passou a ser
**dataset sintético atual (`data/laudos.csv`) vs. Medical Abstracts TC Corpus**.

## Sobre a mudança de target (condição médica em vez de urgência)

A recomendação original do notebook 04 era manter `data/laudos.csv` (único com target de urgência nativo). O
professor autorizou explicitamente, para o Medical Abstracts TC Corpus, aceitar **classificação de condição
médica** (5 classes de especialidade) no lugar de urgência. Isso removeu a objeção principal contra esse
dataset, e a recomendação foi revisada (ver seção final do notebook 04 e
`data/experiments/results/dataset_recommendation.json`): **Medical Abstracts TC Corpus passa a ser o dataset
principal desta linha de experimentação, com `condition_label` como target** — a decisão original (e o
raciocínio completo por trás dela) fica preservada no próprio notebook 04 e em
`dataset_recommendation.json["previous_recommendation"]`, não foi apagada.

Importante: isso é uma decisão para os **experimentos desta branch**. Migrar a API/Airflow/Docker de produção
do target de urgência para condição médica é uma decisão de escopo maior, ainda não tomada — ver notebook 04,
§4 da recomendação revisada.
