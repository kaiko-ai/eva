# *Offline* vs. *online* evaluations

In this tutorial we run *eva* with the three subcommands `predict`, `fit` and `predict_fit`, and take a look at the difference between *offline* and *online* workflows.

For this tutorial we use the [BACH](../../datasets/bach.md) classification task which uses a relatively small dataset.

## *Offline* evaluations

### 1. Compute the embeddings

First, lets use the `predict`-command to download the data and compute embeddings. In this example we use a randomly initialized `dino_vits16` as backbone.

Open a terminal in the folder where you installed *eva* and run:
```
export PRETRAINED=false
export EMBEDDINGS_ROOT=./data/embeddings/dino_vits16_random

python -m eva predict --config configs/vision/dino_vit/offline/bach.yaml
```

Executing this command will:

 - Download and extract the BACH dataset to `./data/bach` (if it has not already been downloaded to this location).
 - Compute the embeddings for all input images with the specified FM-backbone and store them in the `EMBEDDINGS_ROOT` along with a `manifest.csv` file.

Once the session is complete, verify that:

- The raw images have been downloaded to `./data/bach/ICIAR2018_BACH_Challenge`
- The embeddings have been computed and are stored in `./data/embeddings/dino_vits16_random/bach`
- The `manifest.csv` file that maps the filename to the embedding, target and split has been created in `./data/embeddings/dino_vits16/bach`.

### 2. Evaluate the FM 

Now we can use the `fit`-command to evaluate the FM on the precomputed embeddings.

To ensure a quick run for the purpose of this exercise, lets overwrite some of the default parameters. In the terminal, where you run *eva* set:
```
export MAX_STEPS=20
export BATCH_SIZE=100
export LR_VALUE=0.1
```
With a batch size of 100 and a dastaset size of 400 samples, one epoch is equivalent to 400/100=4 steps. Setting `MAX_STEPS` to 20 therefore ensures a maximum of 20/4=5 epochs.

Now fit the decoder classifier, by running
```
python -m eva fit --config configs/vision/dino_vit/offline/bach.yaml
```

Executing this command will:

 - Fit a downstream head (single layer MLP) on the BACH-train split, using the computed embeddings and provided labels as input.
 - Evaluate the trained model on the validation split and store the results.

Once the session is complete:

- Check the evaluation results in `logs/dino_vits16/offline/bach/<session-id>/results.json`. (The `<session-id>` consists of a timestamp and a hash that is based on the run configuration.)
- Take a look at the training curves with the Tensorboard. Open a new terminal, activate the environment and run:
```
tensorboard --logdir logs/dino_vits16/offline/bach
```

### 3. Run a complete *offline*-workflow

With the `predict_fit`-command, the two steps above can be executed with one command. Let's do this now but this time let's use an FM pretrained from ImageNet.

Go back to the terminal and execute:
```
export MAX_STEPS=20
export BATCH_SIZE=100
export LR_VALUE=0.1
export PRETRAINED=true
export EMBEDDINGS_ROOT=./data/embeddings/dino_vits16_pretrained

python -m eva predict_fit --config configs/vision/dino_vit/offline/bach.yaml
```

Once the session is complete, inspect the evaluation results as you did in *Step 2*. Compare the performance metrics and training curves. Can you observe better performance with the ImageNet pretrained encoder?

## *Online* evaluations

Alternatively to the offline workflow from *Step 3*, a complete evaluation can also be computed online. In this case we don't save and track embeddings and instead fit the ML model (encoder with frozen layers + trainable decoder) directly on the given task.

As in *Step 3* above, we again use a `dino_vits16` pretrained from ImageNet. 

Run a complete online workflow with the following command:
```
export MAX_STEPS=20
export BATCH_SIZE=100
export LR_VALUE=0.1
export PRETRAINED=true

python -m eva fit --config configs/vision/dino_vit/online/bach.yaml
```

Executing this command will:

 - Fit a complete model - the frozen FM-backbone and downstream head - on the BACH-train split. (The download step will be skipped if you executed *Step 1* or *3* before.)
 - Evaluate the trained model on the val split and report the results

Once the run is complete:

- Check the evaluation results in `logs/dino_vits16/offline/bach/<session-id>/results.json` and compare them to the results of *Step 3*. Do they match? 
