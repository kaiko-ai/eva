# How to use ***eva***

Before starting to use ***eva***, it's important to get familiar with the different workflows, subcommands and configurations.

## *online* vs. *offline* workflows

We distinguish between the *online* and *offline* workflow:

 - *online*: fits and evaluates a complete model (frozen FM-backbone and trainable head) with images and labels as input.
 - *offline*: separates `predict` (computing of embeddings with FM-backbone) from `fit` (training and evaluation of the decoder).

The *online* workflow can be used to quickly run a complete evaluation without saving and tracking embeddings. The *offline* workflow runs faster (only one FM-backbone forward pass) and is ideal to experiment with different decoders on the same FM-backbone.


## ***eva*** subcommands

To run an evaluation, we call:
```
python -m eva <subcommand> --config <path-to-config-file>
```

The *eva* interface supports the subcommands: `predict`, `fit` and `predict_fit`.

 - **`fit`**: is used to train a decoder for a specific task (e.g. classification) and subsequently evaluate the performance. This can be done *online* (fit directly on input images) or as the 2nd step of the *offline* workflow (fit on input embeddings that were previously computed with the `predict` command)
- **`predict`**: is used to compute embeddings for input images with a provided FM-checkpoint. This is the first step of the *offline* workflow
- **`predict_fit`**: runs `predict` and `fit` sequentially. Like the `fit`-online run, it runs a complete evaluation (both steps of the *offline* workflow) with images as input.


## Run configurations

### Config files

The setup for an ***eva*** run is provided in a `.yaml` config file specified with the `--config` flag.

A config file specifies the setup for the *trainer* (including callback for the model backbone), the *model* (setup of the trainable decoder) and *data* module. 

To get a better understanding, inspect some of the provided [config files](https://github.com/kaiko-ai/eva/tree/main/configs/vision) (which you will download if you run the tutorials).


### Environment variables

To customize runs, you can overwrite some of the config-parameters by setting them as environment variables.

These include:

|                         |                           |
|-------------------------|---------------------------|
| `OUTPUT_ROOT`            | the directory to store logging outputs and recorded results |
| `DINO_BACKBONE`          | the backbone architecture, e.g. "dino_vits16" |
| `PRETRAINED`             | whether to load FM-backbone weights from a pretrained model |
| `MONITOR_METRIC`         | the metric to monitor for early stopping and model checkpoint loading |
| `EMBEDDINGS_DIR`         | the directory to store the computed embeddings |
| `IN_FEATURES`            | the input feature dimension (embedding)           |
| `BATCH_SIZE`             | Batch size for a training step |
| `PREDICT_BATCH_SIZE`             | Batch size for a predict step |