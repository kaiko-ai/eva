# Evaluations with custom models & checkpoints
The `.yaml` evaluation config files that *eva* provides out of the box support loading models from *eva*'s model registry through the `eva.vision.models.ModelFromRegistry` wrapper as described in the [Model Wrapper](./model_wrappers.md) docs.

For evaluating your own custom models & checkpoints, the most flexible way is to create your own set of configs starting from the default ones and replacing the `models: ` section in the `.yaml` file.

However, if your model can be loaded using `timm`, there is a quicker way using the default configuration files:
```
MODEL_NAME=universal/timm_model \
MODEL_EXTRA_KWARGS='{model_name: vit_small_patch16_224.dino, checkpoint_path: path/to/model.ckpt}' \
eva predict_fit --config configs/vision/pathology/offline/segmentation/consep.yaml
```

Note that `MODEL_NAME` in the above example refers to a wrapper model function in *eva*'s model registry which calls `timm.create_model` and therefore can load any `timm` model, while `MODEL_EXTRA_KWARGS.model_name` refers to the name of the model in *timm*`s model registry to be loaded.

