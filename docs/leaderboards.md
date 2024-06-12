---
hide:
  - navigation
---

# Leaderboards

We evaluated the following FMs on the 6 supported WSI-classification tasks. We report *Balanced Accuracy* for binary & multiclass tasks. The score shows the average performance over 5 runs.

<br/>

<center>

| FM-backbone                 | pretraining |  BACH    | CRC       | MHIST     | PCam     |Camelyon16| PANDA    |
|-----------------------------|-------------|--------- |-----------|-----------|----------|----------|----------|
| [DINO ViT-S16](https://arxiv.org/abs/2104.14294) | N/A         | 0.410    | 0.617     | 0.501     | 0.728    | TBD      | TBD      |
| [DINO ViT-S16](https://arxiv.org/abs/2104.14294) | ImageNet    | 0.695    | 0.935     | 0.831     | 0.849    | TBD      | TBD      |
| [Lunit - ViT-S16](https://github.com/lunit-io/benchmark-ssl-pathology/releases/) | TCGA        | 0.801    | 0.934     | 0.768     | 0.895    | TBD      | TBD      |
| [Owkin (Phikon) - iBOT ViT-B16](https://huggingface.co/owkin/phikon) | TCGA        | 0.725    | 0.935     | 0.777     | 0.915    | TBD      | TBD      |
| [UNI - DINOv2 ViT-L16](https://huggingface.co/MahmoodLab/UNI) | Mass-100k   | 0.814    | 0.950     | **0.837** | **0.938**| TBD      | TBD      |
| [kaiko.ai - DINO ViT-S16](https://github.com/kaiko-ai/towards_large_pathology_fms) | TCGA        | 0.797    | 0.943     | 0.828     | 0.893    | TBD      | TBD      |
| [kaiko.ai - DINO ViT-S8](https://github.com/kaiko-ai/towards_large_pathology_fms)	| TCGA        | 0.834    | 0.946     | 0.832     | 0.887    | TBD      | TBD      |
| [kaiko.ai - DINO ViT-B16](https://github.com/kaiko-ai/towards_large_pathology_fms) | TCGA        | 0.810    | **0.960** | 0.826     | 0.898    | TBD      | TBD      |
| [kaiko.ai - DINO ViT-B8](https://github.com/kaiko-ai/towards_large_pathology_fms) | TCGA        | 0.865    | 0.956     | 0.809     | 0.921    | TBD      | TBD      |
| [kaiko.ai - DINOv2 ViT-L14](https://github.com/kaiko-ai/towards_large_pathology_fms) | TCGA        | **0.870**| 0.930     | 0.809     | 0.898    | TBD      | TBD      |

</center>

The runs use the default setup described in the section below.

*eva* trains the decoder on the "train" split and uses the "validation" split for monitoring, early stopping and checkpoint selection. Evaluation results are reported on the "test" split if available and otherwise on the "validation" split.

For details on the FM-backbones and instructions to replicate the results, check out [Replicate evaluations](user-guide/advanced/replicate_evaluations.md).

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
