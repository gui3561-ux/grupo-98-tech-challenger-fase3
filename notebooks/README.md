# Notebooks — EDA e experimentação (branch `feat/eda-dataset`)

Área de **exploração e experimentação**. Nada aqui é código de produção; nada nesta branch altera `src/`,
`Dockerfile`, `docker-compose.yml`, `airflow/` ou `.github/workflows/`. Ver [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md)
nesta pasta para o relatório consolidado (as 15 seções: dataset/modelo escolhidos, justificativa, problemas nos
dados, preprocessing, baseline, modelos testados, resultados, error analysis, comparações, latência,
recomendações e próximos passos).

## Setup

```bash
uv sync --group dev --group experiments
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/<nome>.ipynb   # para reexecutar um notebook
```

## Ordem e dependências entre notebooks

| Notebook | Depende de | Produz |
|---|---|---|
| `01_eda_laudos_sinteticos.ipynb` | `data/laudos.csv` (já no repo) | `data/experiments/results/eda_laudos_summary.json` |
| `02_eda_medical_abstracts.ipynb` | `data/experiments/medical_abstracts/*.parquet` (ver `SOURCE.md` na mesma pasta) | `data/processed/train.csv`, `data/processed/test.csv` (estratégia 02 — target `urgency`) |
| `03_eda_medical_abstracts_second_strategy.ipynb` | `data/experiments/medical_abstracts/*.parquet` | `data/experiments/results/eda_medical_abstracts_summary.json` (estratégia 03 — target `condition_label`) |
| `04_baseline_models.ipynb` | `data/processed/*.csv`, `data/experiments/medical_abstracts/*.parquet` | `data/experiments/results/baseline_results.json` |
| `05_dataset_comparison.ipynb` | `eda_medical_abstracts_summary.json`, `baseline_results.json` | `data/experiments/results/dataset_recommendation.json` |
| `06_model_comparison.ipynb` | `data/processed/*.csv`, `baseline_results.json` | `data/experiments/results/model_comparison_results.json` |
| `07_final_benchmark.ipynb` | `data/processed/*.csv`, `model_comparison_results.json` | `data/experiments/results/latency_optimization_results.json` |

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

## Duas estratégias de target para o Medical Abstracts TC Corpus

O Medical Abstracts TC Corpus não possui nativamente o target de **urgência** (normal/atenção/urgente) exigido
pelo enunciado do Tech Challenge — o target nativo do corpus é `condition_label` (5 classes de especialidade
clínica: neoplasms, digestive system diseases, nervous system diseases, cardiovascular diseases, general
pathological conditions). Para decidir como lidar com essa lacuna, duas estratégias foram exploradas e
comparadas lado a lado, com a mesma configuração de baseline (TF-IDF + RandomForest), para que a diferença de
métrica refletisse a estratégia de dado, não a configuração de modelo:

- **Estratégia 02** (`02_eda_medical_abstracts.ipynb`): mapeia `condition_label` → `urgency` (3 classes),
  batendo literalmente com o target pedido pelo enunciado. Usa um split próprio, estratificado sobre o corpus
  já deduplicado (11.227 abstracts únicos), com verificação explícita de que treino e teste não compartilham
  nenhum texto exato.
- **Estratégia 03** (`03_eda_medical_abstracts_second_strategy.ipynb`): mantém `condition_label` nativo
  (5 classes). Usa o split oficial da fonte, mas quantifica um problema sério nele: ~35% das linhas de teste
  têm texto idêntico ao treino, e **100%** desses pares têm rótulo divergente entre os dois splits — o mesmo
  abstract aparece com uma classe no treino e outra no teste. O notebook `04_baseline_models.ipynb` remove esses
  pares ambíguos do teste antes de medir a performance "honesta" dessa estratégia.

**Resultado da comparação (`04_baseline_models.ipynb` e `05_dataset_comparison.ipynb`):** as duas estratégias
empatam em qualidade de classificação (accuracy ~73%, macro F1 0,71–0,72) — a diferença é menor que 1 ponto
percentual, dentro do ruído esperado entre splits diferentes. A Estratégia 02 treina ~6,5× mais rápido e gera um
modelo quase 2× menor (menos classes, menos linhas de treino), mas a diferença decisiva não é de métrica: é que
a Estratégia 02 entrega o target de urgência pedido literalmente pelo enunciado, sem precisar de nenhuma
exceção, enquanto a Estratégia 03 exigiria aceitar `condition_label` como substituto de urgência.

**Decisão final:** com as métricas empatadas, a equipe optou pela **Estratégia 02 (urgência, 3 classes)** como
target oficial dos experimentos desta branch — ver `data/experiments/results/dataset_recommendation.json`. Os
notebooks `06_model_comparison.ipynb` e `07_final_benchmark.ipynb` usam essa estratégia. Os achados da
Estratégia 03 (vazamento no split oficial, inconsistência de rótulo) continuam documentados e relevantes caso a
equipe reconsidere `condition_label` como target no futuro.

## Fase B — comparação de modelos e otimização (concluída)

- `06_model_comparison.ipynb`: `LogisticRegression`, `LinearSVC` e `MultinomialNB` comparados ao baseline
  RandomForest via validação cruzada (3 folds) no treino + avaliação única no teste, error analysis por classe
  e ablação de preprocessing, tudo sobre a Estratégia 02 (target `urgency`). **Achado principal:**
  `LogisticRegression` vence o baseline em macro F1 (0,7286 vs. 0,7197) sendo ~213× menor e ~39× mais rápido
  nativamente; `LinearSVC` e `MultinomialNB` ficaram abaixo do baseline nesta estratégia. A ablação de
  preprocessing mostrou que `max_features=20000` melhora o macro F1 (0,7362) — diferente do que se observava na
  Estratégia 03, onde vocabulário maior piorava.
- `07_final_benchmark.ipynb`: latência decomposta (preprocessing vs. classificador, cold vs. warm, percentis) e
  conversão ONNX do pipeline completo. **Achado principal:** ONNX continua muito eficaz no RandomForest
  (~158×). No `LogisticRegression` com `max_features=20000` (sem tratamento adicional) o ganho ONNX caiu para
  ~3,2× — vocabulário maior deixa o grafo ONNX proporcionalmente maior. Aplicar `SelectKBest` (chi2, `k=8000`)
  depois do TF-IDF de 20000 features resolve isso **sem contrapartida**: melhora o macro F1 (0,7413, acima até
  da config sem seleção), reduz o arquivo ONNX (644KB vs. 802KB) e reduz a latência ONNX absoluta (0,10ms vs.
  0,12ms) — configuração final recomendada, servida via ONNX (ver `EXPERIMENT_REPORT.md`, seções 11 e 14 para o
  detalhe de por que o número de "speedup" isolado (23×) não conta a história toda).

Ver [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md) para o relatório consolidado com todos os números.
