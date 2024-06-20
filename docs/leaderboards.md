---
hide:
  - navigation
---

# Leaderboards

We evaluated the following FMs on the 6 supported WSI-classification tasks. We report *Balanced Accuracy* for binary & multiclass tasks. The score shows the average performance over 5 runs.

<br/>

<center>

| Vision FM | pretraining | [BACH](datasets/bach.md) | [CRC](datasets/crc.md) | [MHIST](datasets/mhist.md) | [PCam](datasets/patch_camelyon.md) |[Camelyon16](datasets/camelyon16.md)| [PANDA](datasets/panda.md)|
|---------|-------------|--------- |-----------|-----------|----------|----------|----------|
| [DINO ViT-S16](https://arxiv.org/abs/2104.14294) |                                     N/A | 0.411|0.613|0.5|0.752|0.551|0.347|
| [DINO ViT-S16](https://arxiv.org/abs/2104.14294)                                | ImageNet | 0.675|0.936|0.827|0.861|0.751|0.676|
| [Lunit - ViT-S16](https://github.com/lunit-io/benchmark-ssl-pathology/releases/)    | TCGA | 0.77|0.936|0.751|0.905|0.869|0.737|
| [Owkin (Phikon) - iBOT ViT-B16](https://huggingface.co/owkin/phikon)                | TCGA | 0.715|0.942|0.766|0.925|0.879|0.784|
| [UNI - DINOv2 ViT-L16](https://huggingface.co/MahmoodLab/UNI)                  | Mass-100k | 0.797|0.95|0.835|0.939|0.933|0.774|
| [kaiko.ai - DINO ViT-S16](https://github.com/kaiko-ai/towards_large_pathology_fms)  | TCGA | 0.8|0.949|0.831|0.902|0.897|0.77|
| [kaiko.ai - DINO ViT-S8](https://github.com/kaiko-ai/towards_large_pathology_fms)	  | TCGA | 0.825|0.948|0.826|0.887|0.879|0.741|
| [kaiko.ai - DINO ViT-B16](https://github.com/kaiko-ai/towards_large_pathology_fms)  | TCGA | 0.846|0.959|0.839|0.906|0.891|0.753|
| [kaiko.ai - DINO ViT-B8](https://github.com/kaiko-ai/towards_large_pathology_fms)   | TCGA | 0.867|0.952|0.814|0.921|0.939|0.761|
| [kaiko.ai - DINOv2 ViT-L14](https://github.com/kaiko-ai/towards_large_pathology_fms)| TCGA | 0.862|0.935|0.822|0.907|0.941|0.769|

<br/>

![Screenshot](images/starplot.png)

<br/>

</center>

The runs use the default setup described in the section below.

*eva* trains the decoder on the "train" split and uses the "validation" split for monitoring, early stopping and checkpoint selection. Evaluation results are reported on the "test" split if available and otherwise on the "validation" split.

For details on the FM-backbones and instructions to replicate the results, check out [Replicate evaluations](user-guide/advanced/replicate_evaluations.md). For information on the tasks, check out [Datasets](datasets/index.md).

## Evaluation protocol

*eva* uses a task- & model-independent and fixed default set up which closely follows the standard evaluation protocol proposed by [1] (with adjustments for slide-level tasks to ensure convergence and computational efficiency).

We selected this approach to prioritize reliable, robust and fair FM-evaluation while being in line with common literature.

|                                | WSI patch-level tasks     | WSI slide-level tasks     |
|--------------------------------|---------------------------|---------------------------|
| **Backbone**                   | frozen                    | frozen                    |
| **Head**                       | single layer MLP          | ABMIL                     |
| **Dropout**                    | 0.0                       | 0.0                       |
| **Hidden activation function** | n/a                       | ReLU                      |
| **Output activation function** | none                      | none                      |
| **Number of steps**            | 12,500                    | 12,500 (2)                |
| **Base batch size**            | 4,096 (1)                 | 32                        |
| **Base learning rate**         | 0.01 (1)                  | 0.001                     |
| **Early stopping**             | 5% * [Max epochs]         | 10% * [Max epochs] (3)    |
| **Optimizer**                  | SGD                       | AdamW                     |
| **Momentum**                   | 0.9                       | n/a                       |
| **Weight Decay**               | 0.0                       | n/a                       |
| **betas**                      | n/a                       | [0.9, 0.999]              |
| **LR Schedule**                | Cosine without warmup     | Cosine without warmup     |
| **number of patches per slide**| 1                         | dataset specific (4)      |


(1) For smaller datasets (e.g. BACH with 400 samples) we reduce the batch size to 256 and scale the learning rate accordingly.

(2) Upper cap at a maximum of 100 epochs.

(3) Lower cap at a minimum of 8 epochs.

(4) Number of patches per slide depends on task and slide size. For PANDA and Camelyon16 we use a max of 1,000 and 10,000 random patches per slide respectively.


- [1]: [Virchow: A Million-Slide Digital Pathology Foundation Model, 2024](https://arxiv.org/pdf/2309.07778.pdf)
