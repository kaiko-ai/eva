# PathMMU Atlas

PathMMU Atlas is a multimodal visual question answering (VQA) dataset for evaluating vision-language models on pathology knowledge. The Atlas subset uses images from the ARCH Book Set educational pathology textbooks paired with expert-validated multiple-choice questions.

## Raw data

### Key stats
| Modality | Task | Domain | Sample Size | Question Format | License |
|----------|------|--------|-------------|-----------------|---------|
| Image + Text | Multiple Choice (A-E) | Pathology | 80 val / 799 test / 208 test_tiny | VQA with pathology images | CC-BY-ND-4.0 (PathMMU), CC-BY-NC-SA-4.0 (ARCH) |

### Data organization

PathMMU Atlas combines two data sources:

- **PathMMU VQA data**: Expert-validated questions from the [PathMMU HuggingFace repository](https://huggingface.co/datasets/jamessyx/PathMMU)
- **ARCH Book Set images**: 4,270 pathology images from educational textbooks from the [ARCH dataset](https://warwick.ac.uk/fac/cross_fac/tia/data/arch/)

Each sample includes:
- **Image**: A pathology image from the ARCH Book Set
- **Question**: A pathology-related question about the image
- **Options**: Multiple choice answers (A-E)
- **Answer**: The correct option with explanation

Available splits:
- `val`: 80 samples
- `test`: 799 samples
- `test_tiny`: 208 samples

## Download and preprocessing

The dataset can be automatically downloaded by setting `DOWNLOAD_DATA="true"` when running eva. This will download both the ARCH Book Set images and the PathMMU VQA data to the location specified by `DATA_ROOT` (default: `./data/path_mmu_atlas`).

```bash
DOWNLOAD_DATA="true" eva test --config configs/multimodal/pathology/online/multiple_choice/path_mmu_atlas.yaml
```

## Relevant links

- **Paper**: [PathMMU: A Massive Multimodal Expert-Level Benchmark for Understanding and Reasoning in Pathology](https://arxiv.org/abs/2401.16355)
- [**PathMMU HuggingFace**](https://huggingface.co/datasets/jamessyx/PathMMU)
- [**ARCH Dataset**](https://warwick.ac.uk/fac/cross_fac/tia/data/arch/)

## License information

- PathMMU: Released under [CC-BY-ND-4.0](https://creativecommons.org/licenses/by-nd/4.0/)
- ARCH Book Set: Released under [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
