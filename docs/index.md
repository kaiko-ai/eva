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

With the first release, `eva` supports performance evaluation for vision Foundation Models ("FMs") and supervised machine learning ("ML") models on WSI-patch-level image classification- and radiology (CT-scans) segmentation tasks.

The goal of this project is to provide the open-source community with an easy-to-use framework that follows industry practices to provide a robust, reproducible and fair evaluation benchmark across FMs of different sizes and architectures.

Support for additional modalities and tasks will be added in future releases.

## Use cases

### 1. Evaluate your own FMs on public benchmark datasets

With a trained FM as input, you can run `eva` on several publicly available datasets & tasks for which `eva` provides out-of the box support. One `eva` run will automatically download and preprocess the relevant data, compute embeddings with the trained FM, fit and evaluate a classification head and report the mean and standard deviation of the relevant performance metrics the selected task.

Supported datasets & tasks include:

-	**Patch Camelyon**: binary breast cancer classification
-	**BACH**: multiclass breast cancer classification
-	**CRC HE**: multiclass colorectal cancer classification
-	**TotalSegmentator**: radiology/CT-scan for segmentation of anatomical structures

To compare your FM, eva also provides support to evaluate and compare several publicly available models on the same tasks. These include:

-	Pretrained Resnet18 (timm)
-	Baseline FM: DINO with randomly initialized ViT-S16 backbone
-	Lunit: DINO with ViT-S backbone
-	Kaiko: DINO with ViT-S backbone

### 2. Evaluate ML models on your own dataset & task

If you have your own labelled dataset, all that is needed is to implement a dataset class tailored to your source data. Start from one our out-of-the box provided dataset classes, adapt it to your data and run eva to see how different publicly available models are performing on your task.

## Evaluation setup

For WSI-patch-level/microscopy image classification tasks, FMs that produce image embeddings are evaluated with a single linear layer MLP with embeddings as inputs and label-predictions as output.

To standardize evaluations, the default configurations `eva` uses are based on the evaluation protocol proposed by Virchow [1] and dataset/task specific characteristics.

|                         |                           |
|-------------------------|---------------------------|
| **Backbone**            | frozen                    |
| **Hidden layers**       | none                      |
| **Dropout**             | 0.0                       |
| **Activation function** | none                      |
| **Epochs**              | dataset/task specific*    |
| **Batch size**          | dataset/task specific*    |
| **Optimizer**           | SGD                       |
| **Base Learning Rate**  | dataset/task specific*    |
| **Momentum**            | 0.9                       |
| **Weight Decay**        | 0.0                       |
| **Nesterov momentum**   | true                      |
| **LR Schedule**         | Cosine without warmup     |

*The number of epochs, batch size and base learning rate were ran experiments with a pretrained DINO ViT-S16 FM to optimize for convergence with minimal running time, and robust results with repeated runs with different random seeds.

[1]: [Virchow: A Million-Slide Digital Pathology Foundation Model, 2024](https://arxiv.org/pdf/2309.07778.pdf)

## Next steps

Check out the [User Guide](user-guide/index.md) to get started with `eva` 
