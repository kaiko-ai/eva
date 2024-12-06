---
hide:
  - navigation
---

# Leaderboards

We evaluated the following FMs on the 6 supported WSI-classification tasks. We report *Balanced Accuracy* for binary & multiclass tasks and generalized Dice score (no background) for segmentation tasks. The score shows the average performance over 5 runs. Note the leaderboard orders from best to worst according to the average performance across all tasks, excluding BACH (not comparable due to much larger patch size).

<br/>

![Screenshot](images/leaderboard.svg)

<br/>

![Screenshot](images/starplot.png)

<br/>

</center>

The runs use the default setup described in the section below.

*eva* trains the decoder on the "train" split and uses the "validation" split for monitoring, early stopping and checkpoint selection. Evaluation results are reported on the "test" split if available and otherwise on the "validation" split.

For details on the FM-backbones and instructions to replicate the results, check out [Replicate evaluations](user-guide/advanced/replicate_evaluations.md). For information on the tasks, check out [Datasets](datasets/index.md). For Camelyon16 runtime optimization we use only 1000 foreground patches per slide which impacts the performance on this benchmark accross all models. 

## Evaluation protocol

*eva* uses a fixed protocol customized to each category of tasks. The setup has proven to be performant and robust independent of task and model size & architecture and generally prioritizes fairness and comparability over state-of-the-art performance.

We selected this approach to prioritize reliable, robust and fair FM-evaluation while being in line with common literature.

|                                | WSI patch-level classification tasks | WSI slide-level classification tasks | WSI patch-level segmentation tasks |
|--------------------------------|---------------------------|---------------------------|---------------------------|
| **Backbone**                   | frozen                    | frozen                    | frozen                    |
| **Head**                       | single layer MLP          | ABMIL                     | Mult-stage convolutional  |
| **Dropout**                    | 0.0                       | 0.0                       | 0.0                       |
| **Hidden activation function** | n/a                       | ReLU                      | n/a                       |
| **Output activation function** | none                      | none                      | none                      |
| **Number of steps**            | 12,500                    | 12,500 (1)                | 2,000                     |
| **Base batch size**            | 256                       | 32                        | 64                        |
| **Base learning rate**         | 0.0003                    | 0.001                     | 0.002                    |
| **Early stopping**             | 5% * [Max epochs]         | 10% * [Max epochs] (2)    | 10% * [Max epochs] (2)    |
| **Optimizer**                  | SGD                       | AdamW                     | AdamW                     |
| **Momentum**                   | 0.9                       | n/a                       | n/a                       |
| **Weight Decay**               | 0.0                       | n/a                       | n/a                       |
| **betas**                      | n/a                       | [0.9, 0.999]              | [0.9, 0.999]              |
| **LR Schedule**                | Cosine without warmup     | Cosine without warmup     | PolynomialLR              |
| **Loss**                       | Cross entropy             | Cross entropy             | Dice                      |
| **number of patches per slide**| 1                         | dataset specific (3)      | dataset specific (3)      |


(1) Upper cap at a maximum of 100 epochs.

(2) Lower cap at a minimum of 8 epochs.

(3) Number of patches per slide depends on task and slide size. E.g. for `PANDASmall` and `Camelyon16Small` we use a max of 200 and 1000 random patches per slide respectively.
