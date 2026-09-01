# Proveniência — Medical Abstracts TC Corpus

- Dataset original: https://www.kaggle.com/datasets/saharalaa/medical-abstracts-tc-corpus
- Autores/paper: Schopf, Braun & Matthes, "Evaluating Unsupervised Text Classification: Zero-shot and
  Similarity-based Approaches" (NLPIR 2022) — https://doi.org/10.1145/3582768.3582795
- Repositório canônico dos autores: https://github.com/sebischair/Medical-Abstracts-TC-Corpus (license: CC-BY-SA 3.0)

## Por que não foi baixado direto do Kaggle
Não havia CLI/credenciais do Kaggle disponíveis neste ambiente (`kaggle` não instalado, sem `~/.kaggle/kaggle.json`).
O download raw do GitHub (`raw.githubusercontent.com`) retornou `429 Too Many Requests` no momento da coleta.

## Fonte efetivamente usada
Mirror oficial dos mesmos autores no Hugging Face Hub (`TimSchopf/medical_abstracts`), baixado via API pública de
parquet do HF (`/api/datasets/TimSchopf/medical_abstracts/parquet/...`), sem necessidade de autenticação:

- `train.parquet` — 11.550 linhas
- `test.parquet` — 2.888 linhas
- `labels.parquet` — mapeamento condition_label -> condition_name (5 classes)

Contagens por classe conferidas contra a tabela do README oficial do corpus e batem exatamente.
