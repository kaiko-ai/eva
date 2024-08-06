# Model Wrappers


This document shows how to use *eva*'s [Model Wrapper API](../../reference/core/models/networks.md#wrappers) (`eva.models.wrappers`) to load different model formats from a series of sources such as PyTorch Hub, HuggingFace Model Hub and ONNX. 

## Loading PyTorch models
The *eva* framework is built on top of PyTorch Lightning and thus naturally supports loading PyTorch models.
You just need to specify the class path of your model in the backbone section of the `.yaml` config file.

```
backbone:
  class_path: path.to.your.ModelClass
  init_args:
    arg_1: ...
    arg_2: ...
```

Note that your `ModelClass` should subclass `torch.nn.Module` and implement the `forward()` method to return embedding tensors of shape `[embedding_dim]`.

### PyTorch Hub
To load models from PyTorch Hub or other torch model providers, the easiest way is to use the `ModelFromFunction` wrapper class:

```
backbone:
  class_path: eva.models.wrappers.ModelFromFunction
  init_args:
    path: torch.hub.load
    arguments:
      repo_or_dir: facebookresearch/dino:main
      model: dino_vits16
      pretrained: false
    checkpoint_path: path/to/your/checkpoint.torch
```


Note that if a `checkpoint_path` is provided, `ModelFromFunction` will automatically initialize the specified model using the provided weights from that checkpoint file.


### timm
Similar to the above example, we can easily load models using the common vision library `timm`:
```
backbone:
  class_path: eva.models.wrappers.ModelFromFunction
  init_args:
    path: timm.create_model
    arguments:
      model_name: resnet18
      pretrained: true
```


## Loading models from HuggingFace Hub
For loading models from HuggingFace Hub, *eva* provides a custom wrapper class `HuggingFaceModel` which can be used as follows:

```
backbone:
  class_path: eva.models.wrappers.HuggingFaceModel
  init_args:
    model_name_or_path: owkin/phikon
    tensor_transforms: 
      class_path: eva.models.networks.transforms.ExtractCLSFeatures
```

In the above example, the forward pass implemented by the `owkin/phikon` model returns an output tensor containing the hidden states of all input tokens. In order to extract the state corresponding to the CLS token only, we can specify a transformation via the `tensor_transforms` argument which will be applied to the model output.

## Loading ONNX models
`.onnx` model checkpoints can be loaded using the `ONNXModel` wrapper class as follows:

```
class_path: eva.models.wrappers.ONNXModel
init_args:
  path: path/to/model.onnx
  device: cuda
```

## Implementing custom model wrappers

You can also implement your own model wrapper classes, in case your model format is not supported by the wrapper classes that *eva* already provides. To do so, you need to subclass `eva.models.wrappers.BaseModel` and implement the following abstract methods: 

- `load_model`: Returns an instantiated model object & loads pre-trained model weights from a checkpoint if available. 
- `model_forward`: Implements the forward pass of the model and returns the output as a `torch.Tensor` of shape `[embedding_dim]`

You can take the implementations of `ModelFromFunction`, `HuggingFaceModel` and `ONNXModel` wrappers as a reference.
