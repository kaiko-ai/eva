# Train and evaluate a ResNet

If you read [How to use eva](../getting-started/how_to_use.md) and followed the Tutorials to this point, you might ask yourself why you would not always use the *offline* workflow to run a complete evaluation. An *offline*-run stores the computed embeddings and runs faster than the *online*-workflow which computes a backbone-forward pass in every epoch.

One use case for the *online*-workflow is the evaluation of a supervised ML model that does not rely on an backbone/head architecture. To demonstrate this, let's train a ResNet 18 from [Pytoch Image Models (timm)](https://timm.fast.ai/).

To do this we need to create a new config-file:

 - Create a new folder: `configs/vision/resnet18`
 - Create a copy of `configs/vision/dino_vit/online/bach.yaml` and move it to the new folder.

Now let's adapt the new `bach.yaml`-config to the new model:

 - remove the `backbone`-key from the config. If no backbone is specified, the backbone will be skipped during inference.
 - adapt the model-head configuration as follows:

```
     head:
      class_path: eva.models.ModelFromFunction
      init_args:
        path: timm.create_model
        arguments:
          model_name: resnet18
          num_classes: &NUM_CLASSES 4
          drop_rate: 0.0
          pretrained: false
```
To reduce training time, let's overwrite some of the default parameters. Run the training & evaluation with:
```
OUTPUT_ROOT=logs/resnet/bach \
MAX_STEPS=50 \
LR_VALUE=0.01 \
eva fit --config configs/vision/resnet18/bach.yaml
```
Once the run is complete, take a look at the results in `logs/resnet/bach/<session-id>/results.json` and check out the tensorboard with `tensorboard --logdir logs/resnet/bach`. How does the performance compare to the results observed in the previous tutorials?
