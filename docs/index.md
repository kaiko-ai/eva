---
hide:
  - navigation
---

<div align="center">

<img src="./images/eva-logo.png" width="400">

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

With the first release, *eva* supports performance evaluation for vision Foundation Models ("FMs") and supervised machine learning models on WSI-patch-level image classification task. Support for radiology (CT-scans) segmentation tasks will be added soon.

With *eva* we provide the open-source community with an easy-to-use framework that follows industry best practices to deliver a robust, reproducible and fair evaluation benchmark across FMs of different sizes and architectures.

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

-	**[TotalSegmentator](datasets/total_segmentator.md)**: radiology/CT-scan for segmentation of anatomical structures (*support coming soon*)

To evaluate FMs, *eva* provides support for different model-formats, including models trained with PyTorch, models available on HuggingFace and ONNX-models. For other formats custom wrappers can be implemented.


### 2. Evaluate ML models on your own dataset & task

If you have your own labeled dataset, all that is needed is to implement a dataset class tailored to your source data. Start from one of our out-of-the box provided dataset classes, adapt it to your data and run *eva* to see how different FMs perform on your task.

## Evaluation results

We evaluated the following FMs on the 4 supported WSI-patch-level image classification tasks. On the table below we report *Balanced Accuracy* for binary & multiclass tasks and show the average performance & standard deviation over 5 runs.


<center>

| FM-backbone                 | pretraining |  BACH             | CRC                | MHIST              |   PCam/val         | PCam/test       |       
|-----------------------------|-------------|------------------ |-----------------   |-----------------   |-----------------   |--------------     |
| DINO ViT-S16                | N/A         | 0.410 (±0.009)    | 0.617 (±0.008)     | 0.501 (±0.004)     | 0.753 (±0.002)	   | 0.728 (±0.003)    |
| DINO ViT-S16                | ImageNet    | 0.695 (±0.004)    | 0.935 (±0.003)     | 0.831 (±0.002)     | 0.864 (±0.007)     | 0.849 (±0.007)    |
| DINO ViT-B8        	        | ImageNet    | 0.710 (±0.007)    | 0.939 (±0.001)     | 0.814 (±0.003)     | 0.870 (±0.003)     | 0.856 (±0.004)    |
| DINOv2 ViT-L14              | ImageNet    | 0.707 (±0.008)    | 0.916 (±0.002)     | 0.832 (±0.003)     | 0.873 (±0.001)     | 0.888 (±0.001)    |
| Lunit - ViT-S16             | TCGA        | 0.801 (±0.005)    | 0.934 (±0.001)     | 0.768 (±0.004)     | 0.889 (±0.002)     | 0.895 (±0.006)    | 
| Owkin - iBOT ViT-B16        | TCGA        | 0.725 (±0.004)    | 0.935 (±0.001)     | 0.777 (±0.005)     | 0.912 (±0.002)     | 0.915 (±0.003)    | 
| UNI - DINOv2 ViT-L16        | Mass-100k   | 0.814 (±0.008)    | 0.950 (±0.001)     | **0.837 (±0.001)** | **0.936 (±0.001)** | **0.938 (±0.001)**| 
| kaiko.ai - DINO ViT-S16	    | TCGA        | 0.797 (±0.003)    | 0.943 (±0.001)     | 0.828 (±0.003)     | 0.903 (±0.001)     | 0.893 (±0.005)    |
| kaiko.ai - DINO ViT-S8	    | TCGA        | 0.834 (±0.012)    | 0.946 (±0.002)     | 0.832 (±0.006)     | 0.897 (±0.001)     | 0.887 (±0.002)    |
| kaiko.ai - DINO ViT-B16     | TCGA        | 0.810 (±0.008)    | **0.960 (±0.001)** | 0.826 (±0.003)     | 0.900 (±0.002)     | 0.898 (±0.003)    | 
| kaiko.ai - DINO ViT-B8      | TCGA        | 0.865 (±0.019)    | 0.956 (±0.001)     | 0.809 (±0.021)     | 0.913 (±0.001)     | 0.921 (±0.002)  | 
| kaiko.ai - DINOv2 ViT-L14   | TCGA        | **0.870 (±0.005)**| 0.930 (±0.001)     | 0.809 (±0.001)     | 0.908 (±0.001)     | 0.898 (±0.002)    | 

</center>

The runs use the default setup described in the section below.

*eva* trains the decoder on the "train" split and uses the "validation" split for monitoring, early stopping and checkpoint selection. Evaluation results are reported on the "validation" split and, if available, on the "test" split.

For more details on the FM-backbones and instructions to replicate the results, check out [Replicate evaluations](user-guide/advanced/replicate_evaluations.md).

## Evaluation setup

*Note that the current version of eva implements the task- & model-independent and fixed default set up following the standard evaluation protocol proposed by [1] and described in the table below. We selected this approach to prioritize reliable, robust and fair FM-evaluation while being in line with common literature. Additionally, with future versions we are planning to allow the use of cross-validation and hyper-parameter tuning to find the optimal setup to achieve best possible performance on the implemented downstream tasks.*

With a provided FM, *eva* computes embeddings for all input images (WSI patches) which are then used to train a downstream head consisting of a single linear layer in a supervised setup for each of the benchmark datasets. We use early stopping with a patience of 5% of the maximal number of epochs.

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
| **Early stopping**      | 5% * [Max epochs]  |
| **Optimizer**           | SGD                       |
| **Momentum**            | 0.9                       |
| **Weight Decay**        | 0.0                       |
| **Nesterov momentum**   | true                      |
| **LR Schedule**         | Cosine without warmup     |

\* For smaller datasets (e.g. BACH with 400 samples) we reduce the batch size to 256 and scale the learning rate accordingly.

- [1]: [Virchow: A Million-Slide Digital Pathology Foundation Model, 2024](https://arxiv.org/pdf/2309.07778.pdf)

## License

*eva* is distributed under the terms of the [Apache-2.0 license](https://github.com/kaiko-ai/eva?tab=Apache-2.0-1-ov-file#readme).

## Next steps

Check out the [User Guide](user-guide/index.md) to get started with *eva*

<br />

<div align="center">
  <img src="images/kaiko-logo.png" width="200">
</div>
