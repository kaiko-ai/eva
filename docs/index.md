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

With the first open-source release, `eva` supports performance evaluation for vision ML models (FMs and supervised ML models) that are evaluated on WSI-patch-level/microscopy image classification- or radiology CT-scans classification & segmentation tasks.

Support for additional modalities and tasks will be added in future releases.

## Use cases

### 1. Evaluate your own FMs on public benchmark datasets

With only your trained foundation model as input, you can run eva on several publicly available datasets. Out-of the box support (including data download and preprocessing) is supported for popular benchmark. An evaluation run will report the mean and standard deviation of the relevant metrics for each task. Supported datasets/tasks include:

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

For classification tasks, foundation models that produce image embeddings are evaluated with a single linear layer MLP that takes embeddings as inputs and label-predictions as output.

To standardize evaluations, the default configurations eva uses are based on the evaluation protocol proposed by SimCLR and experimentation results.

|                     |                                  |
|---------------------|----------------------------------|
| **Backbone**          | frozen                           |
| **Hidden layers**       | none                             |
| **Dropout**             | 0.0                              |
| **Activation function** | none                             |
| **Epochs**              | [90] task specific experiments*  |
| **Batch size**          | task specific experiments*       |
| **Optimizer**           | SGD                              |
| **Base Learning Rate**  | [0.1] task specific experiments* |
| **Momentum**            | 0.9                              |
| **Weight Decay**        | 0.0                              |
| **Nesterov momentum**   | true                             |
| **LR Schedule**         | Cosine without warmup            |

*We selected the number of epochs, batch size and ran experiments with a pretrained DINO ViT-S16 FM to optimize for convergence with minimal running time, and robust results with repeated runs with different random seeds.

## Next steps

Check out the [User Guide](user-guide/index.md) to get started with `eva` 
