# How to use *eva*

Before starting to use *eva*, it's important to get familiar with the different workflows, subcommands and configurations.


## *eva* subcommands

To run an evaluation, we call:
```
eva <subcommand> --config <path-to-config-file>
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

The setup for an *eva* run is provided in a `.yaml` config file which is defined with the `--config` flag.

A config file specifies the setup for the *trainer* (including callback for the model backbone), the *model* (setup of the trainable decoder) and *data* module. 

The config files for the datasets and models that *eva* supports out of the box, you can find on [GitHub](https://github.com/kaiko-ai/eva/tree/0.0.2). We recommend that you inspect some of them to get a better understanding of their structure and content.


### Environment variables

To customize runs, without the need of creating custom config-files, you can overwrite the config-parameters listed below by setting them as environment variables.

|                         | Type  | Description |
|-------------------------|-------|-------------|
| `OUTPUT_ROOT`           | str   | The directory to store logging outputs and evaluation results |
| `EMBEDDINGS_ROOT`       | str   | The directory to store the computed embeddings |
| `CHECKPOINT_PATH`       | str   | Path to the FM-checkpoint to be evaluated |
| `IN_FEATURES`           | int   | The input feature dimension (embedding) |
| `NUM_CLASSES`           | int   | Number of classes for classification tasks |
| `N_RUNS`                | int   | Number of `fit` runs to perform in a session, defaults to 5 |
| `MAX_STEPS`             | int   | Maximum number of training steps (if early stopping is not triggered) |
| `BATCH_SIZE`            | int   | Batch size for a training step |
| `PREDICT_BATCH_SIZE`    | int   | Batch size for a predict step |
| `LR_VALUE`              | float | Learning rate for training the decoder |
| `MONITOR_METRIC`        | str   | The metric to monitor for early stopping and final model checkpoint loading |
| `MONITOR_METRIC_MODE`   | str   | "min" or "max", depending on the `MONITOR_METRIC` used |
| `REPO_OR_DIR`           | str   | GitHub repo with format containing model implementation, e.g. "facebookresearch/dino:main" |
| `DINO_BACKBONE`         | str   | Backbone model architecture if a facebookresearch/dino FM is evaluated |
| `FORCE_RELOAD`          | bool  | Whether to force a fresh download of the github repo unconditionally |
| `PRETRAINED`            | bool  | Whether to load FM-backbone weights from a pretrained model |
