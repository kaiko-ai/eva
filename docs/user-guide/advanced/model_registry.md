# Backbone Model Registry
*eva* contains a model registry that provides the most popular FM backbones that are publicly available and which we list in the Leaderboard.

## Loading models through the python API
The available models can be listed as follows after installing the *eva* package:
```python
from eva.vision.models.networks.backbones import BackboneModelRegistry

models = BackboneModelRegistry.list_models()
print(models)
```

This should output a list of the model names such as:
```
['universal/vit_small_patch16_224_random', 'pathology/kaiko_vits16', 'pathology/kaiko_vits8', ...]
``` 

A model can then be loaded and instantiated as follows:

```python
import torch
from eva.vision.models.networks.backbones import BackboneModelRegistry

model = BackboneModelRegistry.load_model(
    model_name="universal/vit_small_patch16_224_random",
     **{"out_indices": 2}
)
output = model(torch.randn(1, 3, 224, 224))
print(output.shape)
# console output:
# > torch.Size([1, 384])
```

In the above example, we load a vit-s model initialized with random weights. The `output` tensor corresponds to the CLS embedding which for this backbone is a one dimensional tensor of dimension `384`.
For segmentation tasks, we need to access not only the CLS embedding, but entire feature maps. This we can achieve by using the `out_indices` argument:

```python
model = BackboneModelRegistry.load_model(
    model_name="universal/vit_small_patch16_224_random",
     **{"out_indices": 2}
)
outputs = model(torch.randn(1, 3, 224, 224))
for output in outputs:
    print(output.shape)
# console output:
# > torch.Size([1, 384, 14, 14])
# > torch.Size([1, 384, 14, 14])
```

The above example returns a `list` of 4D tensors, each representing the feature map from a different level in the backbone. `out_indices=2` means that it returns the last two feature maps. This also supports tuples, for instance `(-2, -4)` would return returns the penultimate and the forth before the last maps.


## Run evaluations using backbones from the registry
In the default `.yaml` config files that eva provides, the backbone is specified as follows:

```
backbone:
  class_path: eva.vision.models.wrappers.VisionBackbone
  init_args:
    model_name: ${oc.env:MODEL_NAME, universal/vit_small_patch16_224_imagenet}
    model_kwargs:
      out_indices: ${oc.env:OUT_INDICES, 1}
```

Note that `VisionBackbone` is a model wrapper class, which loads the models through `BackboneModelRegistry`.

By using the `MODEL_NAME` environment variable, you can run an evaluation with a specific model from the registry, without modifying the default config files:
```bash
MODEL_NAME=pathology/kaiko_vits16 \
eva predict_fit --config configs/vision/pathology/offline/segmentation/consep.yaml
```