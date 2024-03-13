# Train and evaluate a ResNet

If you read [How to use eva](../getting-started/how_to_use.md) and followed the Tutorials to this point, you might ask yourself why you would not always use the *offline* workflow to run a complete evaluation. An *offline*-run stores the computed embeddings and runs faster than the *online*-workflow which computes a backbone-forward pass in every epoch.

One use case for the *online*-workflow is the evaluation of a supervised ML model that does not rely on an backbone/head architecture. To demonstrate this, lets train a ResNet 18 from [Pytoch Image Models (timm)](https://timm.fast.ai/).

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
          num_classes: &NUM_CLASSES 2
          drop_rate: 0.0
          pretrained: false
```
To reduce training time, lets overwrite some of the default parameters. In the terminal where you run ***eva***, set:
```
export OUTPUT_ROOT=logs/resnet/bach
export MAX_STEPS=20
export LR_VALUE=0.1
```
Now train and evaluate the model by running:
```
eva fit --config configs/vision/resnet18/bach.yaml
```
Once the run is complete, take a look at the results in `logs/resnet/bach/<session-id>/results.json`. How does the performance compare to the results observed in the previous tutorials?
