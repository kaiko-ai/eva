# How to use *eva*

Before starting to use ***eva***, it's important to get familiar with the different workflows, subcommands and configurations.


## *eva* subcommands

To run an evaluation, we call:
```
python -m eva <subcommand> --config <path-to-config-file>
```

The *eva* interface supports the subcommands: `predict`, `fit` and `predict_fit`.

 - **`fit`**: is used to train a decoder for a specific task and subsequently evaluate the performance. This can be done *online* or *offline* \*
- **`predict`**: is used to compute embeddings for input images with a provided FM-checkpoint. This is the first step of the *offline* workflow
- **`predict_fit`**: runs `predict` and `fit` sequentially. Like the `fit`-online run, it runs a complete evaluation with images as input.

### \* *online* vs. *offline* workflows

We distinguish between the *online* and *offline* workflow:

- *online*: This mode uses raw images as input and generates the embeddings using a frozen FM backbone on the fly to train a downstream head network.
- *offline*: In this mode, embeddings are pre-computed and stored locally in a first step, and loaded in a 2nd step from disk to train the downstream head network.

The *online* workflow can be used to quickly run a complete evaluation without saving and tracking embeddings. The *offline* workflow runs faster (only one FM-backbone forward pass) and is ideal to experiment with different decoders on the same FM-backbone.


## Run configurations

### Config files

The setup for an ***eva*** run is provided in a `.yaml` config file which is defined with the `--config` flag.

A config file specifies the setup for the *trainer* (including callback for the model backbone), the *model* (setup of the trainable decoder) and *data* module. 

To get a better understanding, inspect some of the provided [config files](https://github.com/kaiko-ai/eva/tree/main/configs/vision) (which you will download if you run the tutorials).


### Environment variables

To customize runs, without the need of creating custom config-files, you can overwrite the config-parameters listed below by setting them as environment variables.

|                         |                           |
|-------------------------|---------------------------|
| `OUTPUT_ROOT`            | the directory to store logging outputs and recorded results |
| `DINO_BACKBONE`          | the backbone architecture, e.g. "dino_vits16" |
| `PRETRAINED`             | whether to load FM-backbone weights from a pretrained model |
| `MONITOR_METRIC`         | the metric to monitor for early stopping and model checkpoint loading |
| `EMBEDDINGS_ROOT`        | the directory to store the computed embeddings |
| `IN_FEATURES`            | the input feature dimension (embedding)           |
| `DINO_BACKBONE`          | Backbone model architecture if a facebookresearch/dino FM is evaluated |
| `CHECKPOINT_PATH`        | Path to the FM-checkpoint to be evaluated           |
| `N_RUNS`             | Number of `fit` runs to perform in a session, defaults to 5 |
| `MAX_STEPS`             | Maximum number of training steps (if early stopping is not triggered) |
| `BATCH_SIZE`             | Batch size for a training step |
| `LR_VALUE`             | Learning rate for training the decoder |
| `NUM_CLASSES`             | Number of classes for classification tasks |
| `PREDICT_BATCH_SIZE`             | Batch size for a predict step |
| `MONITOR_METRIC`             | Metric to be monitored for early stopping |
| `MONITOR_METRIC_MODE`             | "min" or "max", depending on the `MONITOR_METRIC` used |