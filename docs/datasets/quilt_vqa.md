# Quilt_VQA

Quilt_VQA is a histopathology visual question answering dataset released with Quilt-LLaVA for evaluating multimodal models on realistic pathology questions. It pairs microscopy frames with naturally occurring questions and answers that were mined from expert-narrated videos and refined with GPT-4 plus manual review.

## Raw data

### Key stats
| Modality | Task | Domain | Sample Size | Question Format | License |
|----------|------|--------|-------------|-----------------|---------|
| Image + Text | Visual Question Answering (free-form) | Histopathology (medical) | 985 evaluation samples | Mix of closed-ended and open-ended questions with short textual answers | CC-BY-NC-ND-3.0 |

### Data organization
- Hugging Face exposes a single `default` configuration with 985 examples stored under a `train` split (eva treats this as the evaluation/test split).
- Each record provides an `image`, `question`, free-form `answer`, categorical `answer_type` (e.g., closed vs. open response), and a short textual `context` snippet from the source narration.
- The repository also packages the original Parquet export (`data/train-*.parquet`) alongside helper files (`quilt_vqa.zip`, `quiltvqa_test_w_ans.json`, `quiltvqa_test_wo_ans.jsonl`) that separate the open and closed subsets used by the Quilt benchmark.

## Download and preprocessing

Quilt_VQA is gated. Accept the terms on the [Hugging Face dataset page](https://huggingface.co/datasets/wisdomik/Quilt_VQA) and generate a user access token before triggering automated downloads.

Once access is granted, set `DOWNLOAD_DATA="true"` (and optionally `DATA_ROOT` for the cache directory) when launching eva with a configuration that references `QuiltVQA`. Provide your Hugging Face token via `HF_TOKEN` so the downloader can authenticate.

## Relevant links

- **Project**: [Quilt-LLaVA](https://quilt-llava.github.io/)
- **Dataset card (Hugging Face)**: https://huggingface.co/datasets/wisdomik/Quilt_VQA
- **Companion dataset**: [Quilt-1M](https://quilt1m.github.io/)
- **Paper**: [Quilt-LLaVA: Visual Instruction Tuning by Extracting Localized Narratives from Open-Source Histopathology Videos](https://arxiv.org/abs/2312.04746)

## License information

Distributed under the [CC-BY-NC-ND 3.0](https://creativecommons.org/licenses/by-nc-nd/3.0/) license. Access is limited to non-commercial research use as outlined in the Hugging Face gated download agreement.
