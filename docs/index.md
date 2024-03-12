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

The goal of this project is to provide the open-source community with an easy-to-use framework that follows industry best practices to provide a robust, reproducible and fair evaluation benchmark across FMs of different sizes and architectures.

Support for additional modalities and tasks will be added in future releases.

## Use cases

### 1. Evaluate your own FMs on public benchmark datasets

With a trained FM as input, you can run ***eva*** on several publicly available datasets & tasks for which ***eva*** provides out-of the box support. One ***eva*** run will automatically download and preprocess the relevant data, compute embeddings with the trained FM, fit and evaluate a classification head and report the mean and standard deviation of the relevant performance metrics the selected task.

Supported datasets & tasks include:

-	**[Patch Camelyon](datasets/patch_camelyon.md)**: binary breast cancer classification
-	**[BACH](datasets/bach.md)**: multiclass breast cancer classification
-	**[CRC](datasets/crc.md)**: multiclass colorectal cancer classification
-	**[MHIST](datasets/mhist.md)**: binary colorectal polyp cancer classification
-	**[TotalSegmentator](datasets/total_segmentator.md)**: radiology/CT-scan for segmentation of anatomical structures

To evaluate FMs, ***eva*** provides support for several formats. These include model checkpoints saved with PyTorch lightning, models available from HuggingFace and onnx-models.


### 2. Evaluate ML models on your own dataset & task

If you have your own labelled dataset, all that is needed is to implement a dataset class tailored to your source data. Start from one our out-of-the box provided dataset classes, adapt it to your data and run eva to see how different publicly available models are performing on your task.

## Evaluation results

We evaluated the following seven FMs on eva on the 4 supported WSI-patch-level image classification tasks:

| FM-backbone                 | pretraining | PCam - val*      | PCam - test*    | BACH - val**    | CRC - val**     | MHIST - val* |
|-----------------------------|-------------|------------------|-----------------|-----------------|-----------------|--------------|
| DINO ViT-S16 random weights | N/A         | 0.765 (±0.0036) | 0.726 (±0.0024) | 0.416 (±0.014)  | 0.643 (±0.0046)	| 0.551 (±0.017)|
| DINO ViT-S16 imagenet       | ImageNet    | 0.871 (±0.0039) | 0.856 (±0.0044) | 0.673 (±0.0041) | 0.936 (±0.0009) | 0.823 (±0.0051)|
| DINO ViT-B8 imagenet	       | ImageNet    | 0.872 (±0.0013) | 0.854 (±0.0015) | 0.704 (±0.008)  | 0.942 (±0.0005) | 0.813 (±0.0026)|
| Lunit - ViT-S16             | TCGA        | 0.89 (±0.0009) | 0.897 (±0.0029) | 0.765 (±0.0108) | 0.936 (±0.001)| 0.762 (±0.0032)| 
| Owkin - iBOT ViT-B16        | TCGA        | 	0.914 (±0.0012) | 0.919 (±0.0082) | 0.717 (±0.0031) | 0.938 (±0.0005)| 0.799 (±0.0021)| 
| kaiko.ai - DINO ViT-S16	    | TCGA        | 0.911 (±0.0017) | 0.899 (±0.002)  | 0.773 (±0.0069) | 0.954 (±0.0012) | 0.829 (±0.0035)|
| kaiko.ai - DINO ViT-B8      | TCGA        | 0.902 (±0.0013) | 0.887 (±0.0031) | 0.798 (±0.0063) | 0.949 (±0.0001) | 0.803 (±0.0038)| 

The reported performance metrics are *balanced binary accuracy* * and *balanced multiclass accuracy* **

The runs used the default setup described in the section below. The table shows the average performance & standard deviation over 5 runs.

***eva*** trains the decoder on the "train" split and uses the "validation" split for monitoring, early stopping and checkpoint selection. Evaluation results are reported on the "validation" split and, if available, on the "test" split.

For more details on the FM-backbones and instructions to replicate those results with ***eva***, refer to the [Replicate results section](user-guide/replicate_evaluations.md) 
in the [User Guide](user-guide/index.md).

## Evaluation setup

For WSI-patch-level/microscopy image classification tasks, FMs that produce image embeddings are evaluated with a single linear layer MLP with embeddings as inputs and label-predictions as output.

To standardize evaluations, the default configurations ***eva*** uses are based on the evaluation protocol proposed by Virchow [1] and dataset/task specific characteristics. To stop training as appropriate we use early stopping after 10% of the maximal number of steps [2].

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
| **Max epochs**          | [n samples] * [Number of steps] /  [Batch size]  |
| **Early stopping**      | 10% * [Max epochs]  |
| **Optimizer**           | SGD                       |
| **Momentum**            | 0.9                       |
| **Weight Decay**        | 0.0                       |
| **Nesterov momentum**   | true                      |
| **LR Schedule**         | Cosine without warmup     |

*For smaller datasets (e.g. BACH with 400 samples) we reduce the batch size to 256 and scale the learning rate accordingly.

- [1]: [Virchow: A Million-Slide Digital Pathology Foundation Model, 2024](https://arxiv.org/pdf/2309.07778.pdf)
- [2]: [Scaling Self-Supervised Learning for Histopathology with Masked Image Modeling](https://www.medrxiv.org/content/10.1101/2023.07.21.23292757v1.full.pdf)

## Next steps

Check out the [User Guide](user-guide/index.md) to get started with ***eva***
