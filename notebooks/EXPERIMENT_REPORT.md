# Relatório de Experimentação — EDA e Seleção de Modelo

**Branch:** `feat/eda-dataset` · **Notebooks:** `notebooks/01` a `notebooks/07`
**Status:** experimentação concluída — aguardando decisão da equipe sobre migração para produção.

Este relatório consolida os notebooks 01–07. Nenhum número aqui é novo: tudo é lido diretamente de
`data/experiments/results/*.json`, gerados pela execução real dos notebooks (não por estimativa).

---

## 1. Dataset e estratégia escolhidos

**Medical Abstracts TC Corpus**, com `condition_label` mapeado para **urgência** (normal/atenção/urgente, 3
classes — "Estratégia 02"), em substituição ao dataset sintético de laudos (`data/laudos.csv`) que a equipe já
usa em produção.

## 2. Justificativa

O Tech Challenge pede um classificador de **urgência** (normal/atenção/urgente), e só o dataset sintético
fornecia esse target nativamente. O Medical Abstracts TC Corpus não tem urgência nativa — seu target nativo é
`condition_label` (5 classes de especialidade clínica). Duas estratégias foram exploradas para resolver essa
lacuna (ver §9 e `notebooks/README.md`):

- **Estratégia 02:** mapear `condition_label` → urgência (3 classes), com split próprio sem leakage.
- **Estratégia 03:** manter `condition_label` nativo (5 classes), usando o split oficial da fonte com os pares
  ambíguos removidos do teste.

As duas estratégias empataram em qualidade de classificação (diferença <1 p.p. em accuracy e macro F1). Com o
empate, o critério decisivo passou a ser adequação ao enunciado: a Estratégia 02 entrega o target de urgência
pedido literalmente, sem precisar de nenhuma exceção de escopo, e ainda treina mais rápido e gera modelo menor.

## 3. Problemas encontrados nos dados

**Dataset sintético (`data/laudos.csv`):**
- 2.000 linhas, mas apenas **107 textos únicos** (diversidade de 5,3%) — gerado por poucos templates × poucas
  keywords por classe.
- O split de treino/teste do pipeline de produção atual tem **99,5%** das linhas de teste com texto idêntico já
  presente no treino (vazamento severo, quantificado no notebook 01).

**Medical Abstracts TC Corpus:**
- **42,5%** das linhas do corpus inteiro estão envolvidas em algum texto duplicado, e a maioria dessas
  duplicatas tem **rótulos diferentes** entre as ocorrências — característica herdada da origem do corpus
  (estilo OHSUMED, onde um abstract pode tocar mais de uma especialidade e foi reduzido a um rótulo único).
- No split oficial treino/teste, **35%** das linhas de teste têm texto idêntico ao treino, e **100%** desses
  pares têm rótulo divergente entre os dois splits (0 de 1.119 pares concordam) — achado da Estratégia 03
  (notebook 03), que motivou a limpeza aplicada no notebook 04.
- A Estratégia 02 evita esse problema por construção: o split é gerado sobre o corpus já deduplicado por texto
  (11.227 abstracts únicos), com verificação explícita (`assert`) de que treino e teste não compartilham
  nenhum texto exato antes de salvar (notebook 02) — 0% de duplicatas dentro do treino, confirmado no
  notebook 06.
- Dois arquivos adicionais investigados e descartados: um derivado do MIMIC-III via Kaggle (`structured_medical
  records`, só 9 internações reais, 100% classe `EMERGENCY`, sem variação) e um "medical_text_classification_
  fake_dataset.csv" (20 textos únicos repetidos 50× cada, classes fabricadas). Nenhum dos dois é usado.
- MIMIC-III (texto livre) e SemClinBr permanecem inacessíveis nesta branch — ambos exigem credenciamento
  humano (PhysioNet DUA / formulário PUCPR) que não foi obtido.

## 4. Estratégia de preprocessing

`TfidfVectorizer(ngram_range=(1,2), stop_words="english")` + modelo — `max_features` testado contra 3 variações
no notebook 06, uma de cada vez, sobre a Estratégia 02:

| Variação | Macro F1 |
|---|---:|
| Baseline (5000 features, bigramas, stopwords en) | 0,7286 |
| Sem remoção de stopwords | 0,7201 |
| Só unigramas | 0,7271 |
| **Vocabulário maior (20000 features)** | **0,7362** |

Ao contrário do que se observava na Estratégia 03 (onde vocabulário maior piorava levemente), aqui aumentar
`max_features` para 20000 **melhora** o resultado. Isso reduz o ganho da otimização ONNX (ver §11) se usado
sozinho — mas o notebook 07 mostra que aplicar `SelectKBest` (chi2) depois do TF-IDF de 20000 features resolve
esse efeito colateral **sem perder o ganho de macro F1** (ver §11/§14): a configuração final usa
`max_features=20000` seguido de `SelectKBest(chi2, k=8000)`, que chega a superar até o macro F1 da config sem
seleção. Remover stopwords continua piorando (vocabulário técnico já é discriminativo).

## 5. Baseline

TF-IDF + RandomForest (`n_estimators=200`), o baseline sugerido como exemplo pelo Tech Challenge, rodado nas
duas estratégias do Medical Abstracts com metodologia idêntica (`Pipeline`, sem vazamento de vocabulário, seed
fixa):

| Estratégia / variante | Accuracy | Macro F1 | Treino (s) | Latência (ms) | Tamanho |
|---|---:|---:|---:|---:|---:|
| Estratégia 03 — split oficial (sem tratamento) | 0,474 | 0,462 | 22,1 | 14,10 | 128,7 MB |
| Estratégia 03 — split limpo (overlap removido) | 0,730 | 0,712 | 22,7 | 14,35 | 128,7 MB |
| **Estratégia 02 — split próprio (sem leakage)** | **0,731** | **0,720** | **3,5** | 14,58 | 65,9 MB |

## 6. Modelos testados

Sobre a Estratégia 02 (target `urgency`): além do baseline RandomForest, `LogisticRegression`, `LinearSVC`,
`MultinomialNB` — os três clássicos leves para texto com TF-IDF, sem necessidade de GPU/tuning pesado.
Validação cruzada estratificada (3 folds) no treino + avaliação única no teste.

## 7. Resultados

| Modelo | Accuracy | Macro F1 | CV Macro F1 | Treino (s) | Latência (ms) | Tamanho |
|---|---:|---:|---:|---:|---:|---:|
| RandomForest (baseline) | 0,731 | 0,720 | — | 3,5 | 14,58 | 65.953 KB |
| **LogisticRegression** | **0,732** | **0,729** | 0,718 ± 0,003 | 2,1 | 0,37 | 310 KB |
| LinearSVC | 0,703 | 0,700 | 0,686 ± 0,010 | 2,0 | 0,41 | 310 KB |
| MultinomialNB | 0,716 | 0,707 | 0,689 ± 0,002 | 1,8 | 0,40 | 427 KB |

**`LogisticRegression` vence o baseline em macro F1 sendo ~213× menor e ~39× mais rápido.** Diferente da
Estratégia 03 (onde os três modelos lineares batiam o RF), aqui só `LogisticRegression` supera o baseline —
`LinearSVC` e `MultinomialNB` ficam abaixo.

## 8. Análise de erros

**Diagnóstico da CV vs. teste:** ao contrário da Estratégia 03 (onde a CV era artificialmente deflacionada por
duplicatas de texto com rótulo inconsistente dentro do treino), a Estratégia 02 não tem esse problema — 0% de
duplicatas dentro do treino (a deduplicação acontece antes do split, no notebook 02). A CV (0,72 / 0,69 / 0,69)
fica próxima do teste, sem gap artificial — confirma que o gap visto na Estratégia 03 era mesmo um artefato de
qualidade de dado do split oficial.

**Erros por classe (melhor modelo, `LogisticRegression`):** as duas classes mais fracas são `normal` (F1=0,610)
e `urgente` (F1=0,774). Diferente do padrão de "convergência para uma classe guarda-chuva" visto na
Estratégia 03, aqui a confusão é **bidirecional entre extremos**: `normal` é confundida principalmente com
`urgente` (156 casos) e `atencao` (130); `urgente` é confundida majoritariamente com `normal` (144 casos) —
quase tanto quanto com `atencao` (31 casos). Isso é um sintoma da natureza do mapeamento condição→urgência:
duas condições médicas com vocabulário clínico parecido podem cair em baldes de urgência opostos dependendo da
regra de mapeamento, então o modelo aprende similaridade textual real, mas essa similaridade nem sempre é
preditiva da urgência atribuída. Não é um bug do classificador — é uma limitação inerente ao próprio
mapeamento condição→urgência, fora do escopo desta branch resolver (revisão de regras de mapeamento).

## 9. Comparação de estratégias (Medical Abstracts)

| Característica | Estratégia 02 (urgência) | Estratégia 03 (condição médica) |
|---|---:|---:|
| Target | urgency (3 classes) | condition_label (5 classes) |
| Origem do split | próprio, sem leakage | oficial, overlap ambíguo removido |
| Linhas de treino | 8.981 | 11.550 |
| Accuracy (split honesto) | 0,731 | 0,730 |
| Macro F1 (split honesto) | 0,720 | 0,712 |
| Tempo de treino baseline (s) | 3,5 | 22,7 |
| Tamanho do modelo baseline | 65,9 MB | 128,7 MB |
| Bate literalmente com o enunciado (urgência)? | Sim | Não (aceito como alternativa) |

**Por que a Estratégia 02 venceu com métricas empatadas:** como a diferença de qualidade de classificação é
estatisticamente desprezível (<1 p.p.), o critério decisivo deixou de ser performance e passou a ser adequação
ao enunciado — a Estratégia 02 entrega urgência nativamente, sem exigir nenhuma exceção de escopo, e ainda é
mais barata computacionalmente (menos classes, menos linhas de treino).

## 10. Comparação de modelos

Ver tabela do §7. Diferente da Estratégia 03 (onde os três modelos lineares batiam o baseline), na
Estratégia 02 só `LogisticRegression` supera o RandomForest em macro F1; `LinearSVC` e `MultinomialNB` ficam
abaixo. A diferença de **tamanho e latência** entre RandomForest e os modelos lineares continua sendo o
critério que mais pesa: mesmo o pior modelo linear (`LinearSVC`, macro F1 0,700) é ~213× menor e ~35× mais
rápido que o baseline, com qualidade só levemente inferior.

## 11. Análise de latência

Decomposição (notebook 07, 200 execuções × 50 amostras, warm):

| Modelo | TF-IDF (ms) | Classificador (ms) | Total (ms) | % no TF-IDF |
|---|---:|---:|---:|---:|
| RandomForest | 0,49 | 13,70 | 14,24 | ~3% |
| LogisticRegression (max_features=20000, sem seleção) | 0,30 | 0,09 | 0,38 | **~79%** |

No `LogisticRegression`, o preprocessing (TF-IDF) domina o tempo, não o classificador. Cold start (1ª chamada
após treinar) é mais lento que warm nos dois modelos (RF: 20,6ms vs. 14,2ms; LogReg: 1,05ms vs. 0,38ms), mas a
diferença absoluta é de poucos milissegundos para o LogReg, sem impacto prático.

**Nativo vs. ONNX** (conversão do pipeline inteiro, mesma técnica do `src/services/onnx_converter.py`):

| Configuração | Nativo (ms) | ONNX (ms) | Speedup | Tamanho nativo | Tamanho ONNX |
|---|---:|---:|---:|---:|---:|
| RandomForest | 14,24 | 0,090 | ~158× | 65.953 KB | 39.161 KB |
| LogisticRegression, sem seleção (max_features=20000) | 0,38 | 0,119 | ~3,2× | 1.278 KB | 802 KB |
| LogisticRegression + `SelectKBest` k=8000 | 2,36* | 0,102 | ~23×* | 1.310 KB | 644 KB |

*\*Ver nota abaixo — o número de "speedup" de `SelectKBest` é maior do que o ganho real de ONNX porque a
versão nativa fica mais lenta, não porque o ONNX melhorou tanto assim.*

**Achado (corrigido de uma rodada anterior deste relatório, que continha um erro de comparação):** com
`max_features=20000` sozinho (sem seleção de features), o ganho de ONNX cai para ~3,2× — bem menor que o
padrão usual — porque o vocabulário maior deixa o grafo ONNX (e a matriz de coeficientes) proporcionalmente
maiores. O arquivo ONNX (802 KB) **continua menor** que o nativo (1.278 KB), só que por margem estreita, não
pelo fator usual.

Testamos a correção proposta (`SelectKBest` chi2 depois do TF-IDF de 20000 features, ver §4) e o resultado
domina a configuração sem seleção em três eixos ao mesmo tempo — macro F1, tamanho e latência ONNX absoluta —
sem nenhuma contrapartida nesses três eixos:

| Configuração | Macro F1 | Tamanho ONNX | Latência ONNX |
|---|---:|---:|---:|
| Sem seleção (20000 features) | 0,7362 | 802 KB | 0,119 ms |
| `SelectKBest` k=8000 | **0,7413** | **644 KB** | **0,102 ms** |

**Ressalva importante sobre o número de "speedup" (23×):** a latência **nativa** (sem ONNX) do pipeline com
`SelectKBest` é ~2,36ms — **~6× mais lenta** que sem seleção (0,38ms), não porque o modelo piorou, mas porque
`SelectKBest.transform()` faz *column slicing* numa matriz esparsa a cada predição individual, com overhead alto
linha a linha no scikit-learn puro. Isso infla o "speedup vs. nativo" sem representar, sozinho, um ganho real de
23×. **O ganho real e sem ressalva é a comparação ONNX-a-ONNX** (0,102ms vs. 0,119ms) e o tamanho de arquivo
(644 KB vs. 802 KB) — ambos genuinamente melhores. Na prática isso significa que `SelectKBest` só compensa se o
modelo for servido via ONNX (que já é a recomendação desta análise); servir nativamente (sem ONNX) com
`SelectKBest` seria pior que sem seleção.

*(faixas — microbenchmarks de latência variam por ruído de máquina entre execuções; ordem de grandeza e
conclusão são estáveis.)*

## 12. Modelo recomendado

**`LogisticRegression`** — vence o baseline RandomForest em macro F1 (0,729 vs. 0,720) sendo ~213× menor e ~39×
mais rápido nativamente, independentemente da escolha de `max_features`.

## 13. Dataset e estratégia recomendados

**Medical Abstracts TC Corpus**, Estratégia 02, com `urgency` (mapeado de `condition_label`) como target — ver
§2 e §9. Limitações que continuam válidas e não devem ser escondidas: modelo maior que o do dataset sintético
mesmo otimizado, corpus em inglês (resto do projeto em português), o mapeamento condição→urgência introduz
ambiguidade real entre classes adjacentes (§8) que não é resolvida trocando de modelo.

## 14. Técnica de otimização recomendada

**Conversão para ONNX Runtime** do pipeline completo continua sendo a escolha certa para o `RandomForest`
(~158× mais rápido). Para o `LogisticRegression`, a recomendação final é `TfidfVectorizer(max_features=20000)`
→ `SelectKBest(chi2, k=8000)` → `LogisticRegression`, **servido via ONNX** — essa configuração não exige
escolher entre qualidade e eficiência: supera a config sem seleção em macro F1 (0,7413 vs. 0,7362), tamanho do
artefato ONNX (644 KB vs. 802 KB) e latência ONNX absoluta (0,102ms vs. 0,119ms), ao mesmo tempo. A única
condição é servir via ONNX — rodar essa configuração nativamente (sem ONNX) seria mais lento que sem seleção,
por causa do overhead de *column slicing* esparso do `SelectKBest` em scikit-learn puro (ver §11). Como ONNX já
é a técnica recomendada de qualquer forma, essa condição não é uma restrição prática. Em qualquer cenário
escolhido: qualquer nova implementação de ONNX deve testar explicitamente que a inferência ONNX concorda com o
modelo nativo nas mesmas entradas antes de ir para produção (risco já identificado em auditoria anterior do
projeto principal, `src/infrastructure/predictors.py`).

## 15. Próximos passos

1. **Decisão da equipe** (não técnica, de produto/escopo): manter a API de produção classificando urgência
   sobre o dataset sintético, migrar para urgência sobre o Medical Abstracts (Estratégia 02, esta
   recomendação), ou manter os dois como linhas paralelas. Nenhuma opção foi implementada nesta branch — só
   investigada e medida.
2. **Configuração de produção** (ver §14): `max_features=20000` + `SelectKBest(chi2, k=8000)`, servida via
   ONNX — domina as alternativas testadas em macro F1, tamanho e latência, sem trade-off. Único cuidado:
   servir via ONNX é parte da recomendação, não opcional (nativo com `SelectKBest` seria mais lento).
3. Se a decisão for migrar: atualizar `src/core/urgency.py` (mapeamento condição→urgência, se ainda não
   nativo), `src/services/trainer.py` (trocar RandomForest por LogisticRegression), `src/services/
   data_generator.py` → substituir por carregamento do Medical Abstracts com o mapeamento da Estratégia 02,
   `airflow/dags/retrain_dag.py`, README e roteiro do vídeo STAR — trabalho de implementação separado, fora do
   escopo desta branch experimental.
4. Se a decisão for manter o dataset sintético: os achados de vazamento e baixa diversidade (notebook 01)
   continuam sendo dívida técnica real e deveriam ser corrigidos independentemente desta decisão de dataset.
5. Corrigir a suíte de testes do projeto principal para testar explicitamente a paridade nativo↔ONNX antes de
   qualquer novo modelo ir para produção (ver §14).
6. Considerar revisar a regra de mapeamento condição→urgência (§8) para reduzir a confusão bidirecional entre
   `normal` e `urgente` — não é um problema de modelo, é um problema de regra de rotulagem.
7. Considerar buscar acesso credenciado a MIMIC-III ou SemClinBr como trabalho futuro, caso a equipe queira um
   corpus clínico real em português — nenhum dos dois foi descartado por qualidade, só por acesso.
