---
hide:
  - navigation
---

<div align="center">

<br />

<img src="./images/eva-logo.png" width="340">

<br />
<br />

<a href="https://pypi.python.org/pypi/kaiko-eva">
  <img src="https://img.shields.io/pypi/v/kaiko-eva.svg?logo=python" />
</a>
<a href="https://github.com/kaiko-ai/eva">
  <img src="https://img.shields.io/badge/repo-main-green?logo=github" />
</a>
<a href="https://github.com/kaiko-ai/eva#license">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg?labelColor=gray" />
</a>

<br />
<br />

</div>

# 

_Oncology FM Evaluation Framework by [kaiko.ai](https://www.kaiko.ai/)_

*eva* currently supports performance evaluation for vision Foundation Models ("FMs") and supervised machine learning models on WSI (patch- and slide-level) as well as radiology image classification tasks.

With *eva* we provide the open-source community with an easy-to-use framework that follows industry best practices to deliver a robust, reproducible and fair evaluation benchmark across FMs of different sizes and architectures.

Support for additional modalities and tasks will be added soon.

## Use cases

### 1. Evaluate your own FMs on public benchmark datasets

With a specified FM as input, you can run *eva* on several publicly available datasets & tasks. One evaluation run will download (if supported) and preprocess the relevant data, compute embeddings, fit and evaluate a downstream head and report the mean and standard deviation of the relevant performance metrics.

Supported datasets & tasks include:

*WSI patch-level pathology datasets*

-	**[Patch Camelyon](datasets/patch_camelyon.md)**: binary breast cancer classification
-	**[BACH](datasets/bach.md)**: multiclass breast cancer classification
-	**[CRC](datasets/crc.md)**: multiclass colorectal cancer classification
-	**[MHIST](datasets/mhist.md)**: binary colorectal polyp cancer classification
- **[MoNuSAC](datasets/monusac.md)**: multi-organ nuclei segmentation
- **[CoNSeP](datasets/consep.md)**: segmentation colorectal nuclei and phenotypes

*WSI slide-level pathology datasets*

-	**[Camelyon16](datasets/camelyon16.md)**: binary breast cancer classification
-	**[PANDA](datasets/panda.md)**: multiclass prostate cancer classification

*Radiology datasets*

-	**[BTCV](datasets/btcv.md)**: Segmentation of abdominal organs (CT scans).
-	**[TotalSegmentator](datasets/total_segmentator.md)**:  Segmentation of anatomical structures (CT scans).
-	**[LiTS](datasets/lits.md)**: Segmentation of liver and tumor (CT scans).

To evaluate FMs, *eva* provides support for different model-formats, including models trained with PyTorch, models available on HuggingFace and ONNX-models. For other formats custom wrappers can be implemented.


### 2. Evaluate ML models on your own dataset & task

If you have your own labeled dataset, all that is needed is to implement a dataset class tailored to your source data. Start from one of our out-of-the box provided dataset classes, adapt it to your data and run *eva* to see how different FMs perform on your task.

## Evaluation results

Check out our [Leaderboards](leaderboards.md) to inspect evaluation results of publicly available FMs.

## License

*eva* is distributed under the terms of the [Apache-2.0 license](https://github.com/kaiko-ai/eva?tab=Apache-2.0-1-ov-file#readme).

## Next steps

Check out the [User Guide](user-guide/index.md) to get started with *eva*

<br />

<div align="center">
  <img src="images/kaiko-logo.png" width="200">
</div>
