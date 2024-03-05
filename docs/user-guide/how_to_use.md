# How to use ***eva***

Before you start using ***eva***, it's important to get familiar with the different workflows, subcommands and configurations which accomodate individual use-cases and user preferences.

## *online* vs. *offline* workflows

To run ***eva***, we distinguish between the *online* and *offline* workflow:

 - *online*: fits and evaluates a complete model (frozen FM-backbone and trainable head) with images and labels as input.
 - *offline*: separates `predict` (computing of embeddings with FM-backbone) from `fit` (training and evaluation of the decoder).

The *online* workflow is the simplest to get started as it does everything in one go, without the overhead of saving and tracking embeddings. The *offline* workflow is faster to run (only one FM-backbone forward pass) and more friendly for experimentation (run different decoder setups on the same computed embeddings).

## ***eva*** subcommands

The *eva* interface supports the commands: `predict`, `fit` and `predict_fit`.

 - **`fit`**: is used to train a decoder for a specific task (e.g. classification) and subsequently evaluate the performance. This can be done *online* (fit directly on input images) or as the 2nd step of the *offline* workflow (fit on input embeddings that were previously computed with the `predict` command)
- **`predict`**: is used to compute embeddings for input images with a provided FM-checkpoint. This is the first step of the *offline* workflow
- **`predict_fit`**: runs `predict` and `fit` sequentially. Like the `fit`-online run, it runs a complete evaluation (both steps of the *offline* workflow) with images as input.

Note that for a complete workflow `predict_fit` will run significantly faster than `fit`. This is because, in the case of `predict_fit`, the FM-forward pass is computed only once, whereas with `fit` the FM-backbone passes through in every training epoch.


### Run the ***eva*** subcommands

To verify that the subcommands work as expected, lets run them on a small subset of git-lfs tracked test-data.

To do this, open a terminal, navigate to the eva root directory and run the `fit` command with:
```
python -m eva fit --config configs/vision/tests/online/patch_camelyon.yaml
```
then, run the `predict` command with:
```
python -m eva predict --config configs/vision/tests/offline/patch_camelyon.yaml
```
and finally run the `predict_fit` command with:
```
python -m eva predict_fit --config configs/vision/tests/offline/patch_camelyon.yaml
```

Each of these commands should complete within a few seconds. This is because with these test-configs we don't download data, only use a small datasample, don't supply an actual FM-backbone and only train for 2 epochs.

To run real examples with the supported datasets and models, proceed to the [tutorials](tutorials.md). 


## Run configurations

As you might have noticed with the example commands above, the setup for an ***eva*** run is provided in a `.yaml` config file that we refer to with the `--config` flag.

A config file specifies the setup for the *trainer* (including callback for the model backbone), the *model* (setup of the trainable decoder) and *data* module. 

To get a better understanding, inspect some of the provided config files in the `config` folder.

To customize runs, some of the parameters can be set through environmental variables. For modifications that exceed the scope of those parameters, it is advised to create a new config file.
