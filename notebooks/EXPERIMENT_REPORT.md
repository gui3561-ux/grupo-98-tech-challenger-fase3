# Relatório de Experimentação — EDA e Seleção de Modelo

**Branch:** `experiment/eda-model-selection` · **Notebooks:** `notebooks/01` a `notebooks/06`
**Status:** experimentação concluída — aguardando decisão da equipe sobre migração para produção.

Este relatório consolida os notebooks 01–06. Nenhum número aqui é novo: tudo é lido diretamente de
`data/experiments/results/*.json`, gerados pela execução real dos notebooks (não por estimativa).

---

## 1. Dataset escolhido

**Medical Abstracts TC Corpus**, com `condition_label` (5 classes de especialidade médica) como target, em
substituição ao dataset sintético de laudos (`data/laudos.csv`) que a equipe já usa em produção.

## 2. Justificativa

A escolha original de dataset partia de uma restrição: o Tech Challenge pede um classificador de **urgência**
(normal/atenção/urgente), e só o dataset sintético fornecia esse target nativamente — por isso o notebook 04
recomendou inicialmente mantê-lo. Essa restrição mudou: consultado sobre a falta de target de urgência no
Medical Abstracts, **o professor autorizou explicitamente aceitar classificação de condição médica no lugar de
urgência para este dataset**. Sem essa restrição, a decisão passou a ser guiada pelos critérios usuais de
qualquer projeto de ML — volume, diversidade, realismo — e nesses três pontos o Medical Abstracts é
objetivamente superior (ver §9).

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
  pares têm rótulo divergente entre os dois splits (0 de 1.119 pares concordam) — verificado por dois métodos
  independentes (notebook 02).
- O mesmo padrão existe **dentro do próprio treino**: 35,2% das linhas de treino envolvidas em duplicata com
  rótulo inconsistente — isso deflaciona validação cruzada feita sem tratamento (ver §6/§10).
- Dois arquivos adicionais investigados e descartados: um derivado do MIMIC-III via Kaggle (`structured_medical
  records`, só 9 internações reais, 100% classe `EMERGENCY`, sem variação) e um "medical_text_classification_
  fake_dataset.csv" (20 textos únicos repetidos 50× cada, classes fabricadas). Nenhum dos dois é usado.
- MIMIC-III (texto livre) e SemClinBr permanecem inacessíveis nesta branch — ambos exigem credenciamento
  humano (PhysioNet DUA / formulário PUCPR) que não foi obtido.

## 4. Estratégia de preprocessing

`TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english")` + modelo — testado contra 3
variações no notebook 05, uma de cada vez:

| Variação | Macro F1 |
|---|---:|
| Baseline (5000 features, bigramas, stopwords en) | **0,7204** |
| Sem remoção de stopwords | 0,6977 |
| Só unigramas | 0,7194 |
| Vocabulário maior (20000 features) | 0,7160 |

Remover stopwords **piora** o resultado (vocabulário técnico já é discriminativo); aumentar `max_features`
também piora levemente e custa mais espaço. A configuração original já é a melhor testada — decisão baseada
nos números, não em prática "geralmente recomendada".

## 5. Baseline

TF-IDF + RandomForest (`n_estimators=200`), o baseline sugerido como exemplo pelo Tech Challenge, rodado nos
dois datasets com metodologia idêntica (`Pipeline`, sem vazamento de vocabulário, seed fixa):

| Dataset / variante | Accuracy | Macro F1 | Treino (s) | Latência (ms) | Tamanho |
|---|---:|---:|---:|---:|---:|
| Laudos — split ingênuo (produção atual) | 1,000 | 1,000 | 0,19 | 13,8 | 599 KB |
| Laudos — split deduplicado (honesto) | 1,000 | 1,000 | 0,13 | 13,8 | 520 KB |
| Medical Abstracts — split oficial | 0,474 | 0,462 | 21,0 | 13,8 | 125,7 MB |
| Medical Abstracts — split limpo (honesto) | 0,730 | 0,712 | 20,9 | 13,8 | 125,7 MB |

## 6. Modelos testados

Além do baseline RandomForest: `LogisticRegression`, `LinearSVC`, `MultinomialNB` — os três clássicos leves
para texto com TF-IDF, sem necessidade de GPU/tuning pesado. Validação cruzada estratificada (3 folds) no
treino + avaliação única no teste limpo.

## 7. Resultados

| Modelo | Accuracy | Macro F1 | CV Macro F1* | Treino (s) | Latência (ms) | Tamanho |
|---|---:|---:|---:|---:|---:|---:|
| RandomForest (baseline) | 0,730 | 0,712 | — | 20,9 | 13,84 | 125,7 MB |
| **LogisticRegression** | **0,727** | **0,720** | 0,566 ± 0,006 | 2,9 | 0,39 | 389 KB |
| LinearSVC | 0,721 | 0,717 | 0,518 ± 0,004 | 2,6 | 0,40 | 389 KB |
| MultinomialNB | 0,699 | 0,678 | 0,545 ± 0,011 | 2,2 | 0,41 | 584 KB |

*\*A CV é sistematicamente mais baixa que o teste — ver §8, é um artefato de dado, não de modelo.*

**`LogisticRegression` vence o baseline em macro F1 sendo 331× menor e ~35× mais rápido.**

## 8. Análise de erros

**Diagnóstico da CV vs. teste:** a CV deu macro F1 na faixa de 0,52–0,57, bem abaixo dos 0,68–0,72 do teste.
Investigado (não apenas reportado): 35,2% das linhas de treino têm duplicata de texto com rótulo inconsistente
dentro do próprio treino — quando um desses pares cai um de cada lado de um fold da CV, o modelo é penalizado
por "errar" em um caso onde a fonte se contradiz. O teste limpo já exclui esse tipo de ambiguidade (é por isso
que existe), então o macro F1 do teste é a estimativa confiável; a CV serve só para ver a variância (desvio
padrão), não o nível absoluto.

**Erros por classe (melhor modelo, `LogisticRegression`):** as duas classes mais fracas são `nervous system
diseases` (F1=0,64) e `general pathological conditions` (F1=0,66). Os falsos negativos de `nervous system
diseases` se concentram majoritariamente em `general pathological conditions` (86 de 112 casos) — a classe mais
genérica do corpus funciona quase como "outros" dentro do esquema de 5 categorias, absorvendo casos ambíguos de
várias especialidades. Exemplos reais de erro (notebook 05) mostram abstracts que genuinamente tocam mais de
uma especialidade (ex.: um caso sobre tumor neuroendócrino classificado como `nervous system diseases` mas
previsto como `neoplasms`) — sobreposição semântica real, não um bug do classificador. Trocar de modelo não
resolve isso; resolveria só com revisão do esquema de classes, fora do escopo desta branch.

## 9. Comparação de datasets

| Característica | Laudos sintéticos (atual) | Medical Abstracts TC Corpus |
|---|---:|---:|
| Registros | 2.000 | 14.438 |
| Textos únicos | 107 (5,3%) | 11.227 (77,8%) |
| Classes | normal, atencao, urgente | 5 (especialidade clínica) |
| Target nativo é urgência? | Sim | Não |
| Desbalanceamento | 2,5× | 3,2× |
| Texto médio (palavras) | 5,4 | 179,9 |
| Vazamento treino/teste | 99,5% do teste | 35,0% do teste (rótulo sempre diverge) |
| Accuracy baseline (honesto) | 1,000 (trivial) | 0,730 (real) |
| Licenciamento | Total (interno) | CC-BY-SA 3.0 |

**Por que escolher o dataset com números menores:** 100% no dataset sintético não é evidência de um modelo bom
— é evidência de um problema trivial (vocabulário quase exclusivo por classe, decoreba). 73% no Medical
Abstracts vem da mesma metodologia aplicada a texto real — a queda de 100% para 73% é inteiramente explicada
pela diferença de dado, não de modelo. Entre uma métrica alta mas vazia e uma métrica moderada mas real, a
decisão é pela métrica real (notebook 04, detalhado).

## 10. Comparação de modelos

Ver tabela do §7. `LogisticRegression` e `LinearSVC` ficam próximos em todas as métricas; `MultinomialNB` é o
mais barato de treinar mas tem a pior macro F1. A diferença de qualidade entre os 4 modelos é pequena — o
gargalo é sobreposição real de vocabulário entre classes (§8), não capacidade do algoritmo. A diferença de
**tamanho e latência** entre RandomForest e os três modelos lineares é o critério que mais pesa na escolha
final.

## 11. Análise de latência

Decomposição (notebook 06, 200 execuções × 50 amostras, warm):

| Modelo | TF-IDF (ms) | Classificador (ms) | Total (ms) | % no TF-IDF |
|---|---:|---:|---:|---:|
| RandomForest | 0,42–0,45 | 13,3–13,4 | 13,7–13,9 | ~3% |
| LogisticRegression | 0,29 | 0,09 | 0,39 | **~74%** |

No `LogisticRegression`, o preprocessing (TF-IDF) domina o tempo, não o classificador — informação relevante
para onde investir otimização futura. Cold start (1ª chamada após treinar) é mais lento que warm nos dois
modelos, mas a diferença absoluta é de poucos milissegundos, sem impacto prático.

**Nativo vs. ONNX** (conversão do pipeline inteiro, mesma técnica do `src/services/onnx_converter.py`):

| Modelo | Nativo (ms) | ONNX (ms) | Speedup | Tamanho nativo | Tamanho ONNX |
|---|---:|---:|---:|---:|---:|
| RandomForest | 13,7–13,9 | 0,10–0,14 | ~90–140× | 125,7 MB | 78,1 MB |
| LogisticRegression | 0,39 | 0,06 | ~6–7× | 389 KB | 238 KB |

*(faixas — microbenchmarks de latência variam por ruído de máquina entre execuções; ordem de grandeza e
conclusão são estáveis.)*

## 12. Modelo recomendado

**`LogisticRegression`** — vence o baseline RandomForest em macro F1 (0,720 vs. 0,712) sendo 331× menor e ~35×
mais rápido nativamente, e ainda ganha mais alguns múltiplos de velocidade com ONNX. `LinearSVC` é um segundo
candidato válido, muito próximo em todas as métricas.

## 13. Dataset recomendado

**Medical Abstracts TC Corpus**, com `condition_label` como target — ver §2 e §9. Limitações que continuam
válidas e não devem ser escondidas: modelo maior que o do dataset sintético mesmo otimizado, corpus em inglês
(resto do projeto em português), split oficial não deve ser usado sem a limpeza validada no notebook 03/05/06.

## 14. Técnica de otimização recomendada

**Conversão para ONNX Runtime** do pipeline completo (TF-IDF + classificador), reaproveitando o
`ONNXConverter` já existente em produção. Ganho real mesmo em um modelo já pequeno/rápido (~6–7×, ~85% de
redução de latência) porque a conversão otimiza o TF-IDF junto com o classificador — não só o modelo. Risco
identificado e documentado: a auditoria do projeto principal já encontrou um bug real de mapeamento de classes
na conversão ONNX atual (`src/infrastructure/predictors.py`) — qualquer nova implementação de ONNX deve testar
explicitamente que a inferência ONNX concorda com o modelo nativo nas mesmas entradas antes de ir para produção.

## 15. Próximos passos

1. **Decisão da equipe** (não técnica, de produto/escopo): manter a API de produção classificando urgência
   sobre o dataset sintético, migrar para condição médica sobre o Medical Abstracts, ou manter os dois como
   linhas paralelas. Nenhuma opção foi implementada nesta branch — só investigada e medida.
2. Se a decisão for migrar: atualizar `src/core/urgency.py` (enum de classes), `src/services/trainer.py`
   (trocar RandomForest por LogisticRegression), `src/services/data_generator.py` → substituir por carregamento
   do Medical Abstracts, `airflow/dags/retrain_dag.py`, README e roteiro do vídeo STAR — trabalho de
   implementação separado, fora do escopo desta branch experimental.
3. Se a decisão for manter urgência: os achados de vazamento e baixa diversidade do dataset sintético
   (notebook 01) continuam sendo dívida técnica real e deveriam ser corrigidos independentemente desta decisão
   de dataset (já documentado na auditoria técnica anterior do projeto).
4. Corrigir a suíte de testes do projeto principal para testar explicitamente a paridade nativo↔ONNX antes de
   qualquer novo modelo ir para produção (ver §14).
5. Considerar buscar acesso credenciado a MIMIC-III ou SemClinBr como trabalho futuro, caso a equipe queira um
   corpus clínico real em português — nenhum dos dois foi descartado por qualidade, só por acesso.
