# How to use eva

## *online* vs *offline*

To run ***eva*** we distinguish between the *online* and *offline* workflow:

 - *online*: fits and evaluates a complete model (frozen FM-backbone and trainable head) with images and labels as input.
 - *offline*: separates `predict` (computing of embeddings with FM-backbone) from `fit` (training and evaluation of the decoder).

The *online* workflow is the simplest to get started as it does everything in one go, without the overhead of saving and tracking embeddings. The *offline* workflow is faster to run (only one FM-backbone forward pass) and more friendly for experimentation (run different decoder setups on the same computed embeddings).

## The ***eva*** subcommands

The *eva* interface supports the commands: `predict`, `fit` and `predict_fit`.

 - **`fit`**: is used to train a decoder for a specific task (e.g. classification) and subsequently evaluate the performance. This can be done *online* (fit directly on input images) or as the 2nd step of the *offline* workflow (fit on input embeddings that were previously computed with the `predict` command)
- **`predict`**: is used to compute embeddings for input images with a provided FM-checkpoint. This is the first step of the *offline* workflow
- **`predict_fit`**: runs `predict` and `fit` sequentially. Like the `fit`-online run, it runs a complete evaluation (both steps of the *offline* workflow) with images as input.

Note that for a complete workflow `predict_fit` will run significantly faster than `fit`. This is because, in the case of `predict_fir`, the FM-forward pass is computed only once, whereas with `fit` the FM-backbone passes through in every training epoch.


### Run the ***eva*** subcommands

To verify that the subcommands work as expected, lets run them on a small subset of git-lfs tracked test-data.

Now open a terminal, navigate to the eva root directory and run the `fit` command with:
```
python -m eva fit --config configs/vision/tests/online/patch_camelyon.yaml
```
the `predict` command with:
```
python -m eva predict --config configs/vision/tests/offline/patch_camelyon.yaml
```
and the `predict_fit` command with:
```
python -m eva predict_fit --config configs/vision/tests/offline/patch_camelyon.yaml
```

Each of these commands should complete within a few seconds. This is because with these test-configs we don't download data, only use a small datasample, don't supply an actual FM-backbone and only train for 2 epochs.

To run real examples with the supported datasets and models, proceed to the [tutorials](tutorials.md). 
