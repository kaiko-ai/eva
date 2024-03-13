---
hide:
  - navigation
---

<div align="center">

<img src="./images/eva-logo.png" width="400">

<br />


<a href="https://www.python.org/">
  <img src="https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white" />
</a>
<a href="https://www.apache.org/licenses/LICENSE-2.0">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" />
</a>

<br />

<p align="center">
  <a href="user-guide">User Guide</a> •
  <a href="datasets">Datasets</a> •
  <a href="reference">Reference API</a>
</p>

</div>

_Oncology FM Evaluation Framework by Kaiko_

With the first release, ***eva*** supports performance evaluation for vision Foundation Models ("FMs") and supervised machine learning ("ML") models on WSI-patch-level image classification- and radiology (CT-scans) segmentation tasks.

The goal of this project is to provide the open-source community with an easy-to-use framework that follows industry best practices to deliver a robust, reproducible and fair evaluation benchmark across FMs of different sizes and architectures.

Support for additional modalities and tasks will be added in future releases.

## Use cases

### 1. Evaluate your own FMs on public benchmark datasets

With a specified FM as input, you can run *eva* on several publicly available datasets & tasks. One evaluation run will download and preprocess the relevant data, compute embeddings, fit and evaluate a downstream head and report the mean and standard deviation of the relevant performance metrics.

Supported datasets & tasks include:

*WSI patch-level pathology datasets*

-	**[Patch Camelyon](datasets/patch_camelyon.md)**: binary breast cancer classification
-	**[BACH](datasets/bach.md)**: multiclass breast cancer classification
-	**[CRC](datasets/crc.md)**: multiclass colorectal cancer classification
-	**[MHIST](datasets/mhist.md)**: binary colorectal polyp cancer classification

*Radiology datasets*

-	**[TotalSegmentator](datasets/total_segmentator.md)**: radiology/CT-scan for segmentation of anatomical structures

More datasets & downstream task types will be added in future releases.

To evaluate FMs, *eva* provides support for different model-formats, including models trained with PyTorch, models available on HuggingFace and ONNX-models. For other formats custom wrappers can be implemented.


### 2. Evaluate ML models on your own dataset & task

If you have your own labelled dataset, all that is needed is to implement a dataset class tailored to your source data. Start from one of our out-of-the box provided dataset classes, adapt it to your data and run *eva* to see how different FMs perform on your task.

## Evaluation results

We evaluated the following FMs on the 4 supported WSI-patch-level image classification tasks:

| FM-backbone                 | pretraining | PCam - val*      | PCam - test*    | BACH - val**    | CRC - val**     | MHIST - val* |
|-----------------------------|-------------|------------------|-----------------|-----------------|-----------------|--------------|
| DINO ViT-S16                | N/A         | 0.765 (±0.004) | 0.726 (±0.003) | 0.416 (±0.014)  | 0.643 (±0.005)	| 0.551 (±0.017)|
| DINO ViT-S16                | ImageNet    | 0.871 (±0.004) | 0.856 (±0.005) | 0.673 (±0.005) | 0.936 (±0.001) | 0.823 (±0.006)|
| DINO ViT-B8        	        | ImageNet    | 0.872 (±0.004) | 0.854 (±0.002) | 0.704 (±0.008)  | 0.942 (±0.001) | 0.813 (±0.003)|
| Lunit - ViT-S16             | TCGA        | 0.89 (±0.001) | 0.897 (±0.003) | 0.765 (±0.011) | 0.936 (±0.001)| 0.762 (±0.004)| 
| Owkin - iBOT ViT-B16        | TCGA        | 	**0.914 (±0.002)** | **0.919 (±0.009)** | 0.717 (±0.004) | 0.938 (±0.001)| 0.799 (±0.003)| 
| kaiko.ai - DINO ViT-S16	    | TCGA        | 0.911 (±0.002) | 0.899 (±0.002)  | **0.773 (±0.007)** | **0.954 (±0.002)** | **0.829 (±0.004)**|
| kaiko.ai - DINO ViT-B8      | TCGA        | 0.902 (±0.002) | 0.887 (±0.004) | 0.798 (±0.007) | 0.949 (±0.001) | 0.803 (±0.004)| 

\* Metric in table: *Balanced Accuracy* (for binary & multiclass). The runs use the default setup described in the section below. The table shows the average performance & standard deviation over 5 runs.

***eva*** trains the decoder on the "train" split and uses the "validation" split for monitoring, early stopping and checkpoint selection. Evaluation results are reported on the "validation" split and, if available, on the "test" split.

For more details on the FM-backbones and instructions to replicate the results, please refer to the [Replicate evaluations](user-guide/advanced/replicate_evaluations.md).

## Evaluation setup

### WSI-patch-level image classification tasks
With the FM we generate embeddings for all WSI patches and then use these embeddings as input to train a downstream head consisting of a single linear layer in a supervised setup for each of the benchmark datasets. The FM weights are frozen throughout this process.

To standardize evaluations, the default configurations *eva* uses are based on the evaluation protocol proposed by [1] and dataset/task specific characteristics. We use early stopping after 10% of the maximal number of steps as suggested by [2].

|                         |                           |
|-------------------------|---------------------------|
| **Backbone**            | frozen                    |
| **Hidden layers**       | none                      |
| **Dropout**             | 0.0                       |
| **Activation function** | none                      |
| **Number of steps**     | 12,500                    |
| **Base Batch size**     | 4,096                     |
| **Batch size**          | dataset specific*         |
| **Base learning rate**  | 0.01                      |
| **Learning Rate**       | [Base learning rate] * [Batch size] / [Base batch size]   |
| **Max epochs**          | [Number of samples] * [Number of steps] /  [Batch size]  |
| **Early stopping**      | 10% * [Max epochs]  |
| **Optimizer**           | SGD                       |
| **Momentum**            | 0.9                       |
| **Weight Decay**        | 0.0                       |
| **Nesterov momentum**   | true                      |
| **LR Schedule**         | Cosine without warmup     |

\* For smaller datasets (e.g. BACH with 400 samples) we reduce the batch size to 256 and scale the learning rate accordingly.

- [1]: [Virchow: A Million-Slide Digital Pathology Foundation Model, 2024](https://arxiv.org/pdf/2309.07778.pdf)
- [2]: [Scaling Self-Supervised Learning for Histopathology with Masked Image Modeling](https://www.medrxiv.org/content/10.1101/2023.07.21.23292757v1.full.pdf)

## License

*eva* is distributed under the terms of the [Apache-2.0 license](https://github.com/kaiko-ai/eva?tab=Apache-2.0-1-ov-file#readme).

## Next steps

Check out the [User Guide](user-guide/index.md) to get started with *eva*

<br />

<div align="center">
  <img src="images/kaiko-logo.png" width="200">
</div>
