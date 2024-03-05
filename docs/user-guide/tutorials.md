# Tutorials

## 1. Run the three commands

To quickly see how the eva commands work, let's run each of them on a small subset of git-lfs tracked test-data.

### `fit`

First lets run a complete online-evaluation workflow on a small subset of patch-camelyon data. For this we'll use the 
config `configs/vision/tests/online/patch_camelyon.yaml`. Open it to see what it does.

You might note that with this test config:

 - We only use a few sample images (instead of the complete dataset with >300k samples)
 - We don't download the data, since we have it git-lfs tracked and stored in `tests/eva/assets/vision/datasets/patch_camelyon`
 - we train for only 2 epochs
 - we don't use a FM-backbone, but simply flatten the input image

Now open a terminal, navigate to the eva root directory and run:
```
python -m eva fit --config configs/vision/tests/online/patch_camelyon.yaml
```
This will run a complete evaluation in a few seconds. Navigate to `logs/dino_vits16/patch_camelyon`[TBD] where the evaluation results are stored and check the output of the evaluation run.

### `predict`

To only compute the embeddings, we run the `fit` command with the config


## 2. Run a complete online-evaluation with `fit`


we use the [BACH](../datasets/bach.md) dataset