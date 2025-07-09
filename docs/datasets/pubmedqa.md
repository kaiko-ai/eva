# PubMedQA

PubMedQA is a biomedical question-answering dataset for evaluating large language models on medical knowledge. The task requires models to classify answers as "yes", "no", or "maybe" based on biomedical questions paired with abstracts from PubMed.

## Raw data

### Key stats
| Modality | Task | Domain | Sample Size | Question Format | License |
|----------|------|--------|-------------|-----------------|---------|
| Text | Classification (3 classes) | Biomedical | 1,000 manually annotated test samples | Medical Q&A with abstracts | MIT License |

### Data organization

PubMedQA is split into three subsets: PQA-A(rtificial), PQA-U(nlabeled) and PQA-L(abeled).

- **PQA-L(abeled)**: 1,000 manually curated question-abstract-answer triplets with expert annotations (used by eva)
- **PQA-A(rtificial)**: 55k artificially generated samples (not used in eva)
- **PQA-U(nlabeled)**: 211k questions without gold standard answers (not used in eva)

Each sample includes:
- **Question**: A biomedical research question
- **Context**: Relevant PubMed abstract(s)
- **Answer**: Expert-annotated classification ("yes", "no", "maybe")

## Download and preprocessing

The dataset can be automatically downloaded by setting `DOWNLOAD_DATA="true"` when running eva. The data will be downloaded to the location specified by `DATA_ROOT` (default: `./data/pubmedqa`).

```bash
DOWNLOAD_DATA="true" eva validate --config configs/language/pubmedqa.yaml
```

## Relevant links

- **Paper**: [PubMedQA: A Dataset for Biomedical Research Question Answering](https://arxiv.org/abs/1909.06146)
- [**Official Repository**](https://github.com/pubmedqa/pubmedqa)
- [**Leaderboard**](https://pubmedqa.github.io/)
- [**HuggingFace**](https://huggingface.co/datasets/bigbio/pubmed_qa)

## License information

Released under the [MIT License](https://github.com/pubmedqa/pubmedqa/blob/master/LICENSE)