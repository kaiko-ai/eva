# Model Wrappers


This document shows how to use *eva*'s [Model Wrapper API](../../reference/core/models/networks.md#wrappers) (`eva.models.wrappers`) to load different model formats from a series of sources such as PyTorch Hub, HuggingFace Model Hub and ONNX.

## *eva* model registry
To load models from *eva*'s FM backbone [model registry](./model_registry.md), we provide the `ModelFromRegistry` wrapper class:

```
backbone:
  class_path: eva.vision.models.wrappers.ModelFromRegistry
  init_args:
    model_name: universal/vit_small_patch16_224_dino
    model_kwargs:
      out_indices: 1
```
The above example loads a vit-s model with weights pretrained on imagenet-1k. Note that by specifying the `out_indices=1` keyword argument, the model will return a feature map tensor, which is needed for segmentation tasks. If you ommit this argument, it will return the CLS embedding (for classification tasks).

## PyTorch models
The *eva* framework is built on top of PyTorch Lightning and thus naturally supports loading PyTorch models.
You just need to specify the class path of your model in the backbone section of the `.yaml` config file.

```
backbone:
  class_path: path.to.your.ModelClass
  init_args:
    arg_1: ...
    arg_2: ...
```

Note that your `ModelClass` should subclass `torch.nn.Module` and implement the `forward()` method to return an embedding tensor of shape `[1, embedding_dim]` for classification tasks or a list feature maps of shape `[1, embedding_dim, patch_dim, patch_dim]` for segmentation.

## Models from functions
The wrapper class `eva.models.wrappers.ModelFromFunction` allows you to load models from Python functions that return torch model instances (`nn.Module`).

You can either use this to load models from your own custom functions, or from public providers such as Torch Hub or `timm` that expose model load functions.

### `torch.hub.load`
The following example shows how to load a dino_vits16 model from Torch Hub using the `torch.hub.load` function:
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

### `timm.create_model`
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

## `timm` models
While you can load `timm` models using the `ModelFromFunction` wrapper class as shown in the example above, we also provide a specific wrapper class:

```
backbone:
  class_path: eva.vision.models.wrappers.TimmModel
  init_args:
    model_name: vit_tiny_patch16_224
    pretrained: true
    out_indices=1 # to return the last feature map
    model_kwargs:
      dynamic_img_size: true  

```

## HuggingFace models
For loading models from HuggingFace Hub, *eva* provides a custom wrapper class `HuggingFaceModel` which can be used as follows:

```
backbone:
  class_path: eva.models.wrappers.HuggingFaceModel
  init_args:
    model_name_or_path: owkin/phikon
    tensor_transforms: 
      class_path: eva.models.networks.transforms.ExtractCLSFeatures
```

In the above example, the forward pass implemented by the `owkin/phikon` model returns an output tensor containing the hidden states of all input tokens. In order to extract the state corresponding to the CLS token only (for classification tasks), we can specify a transformation via the `tensor_transforms` argument which will be applied to the model output. For segmentation tasks, we can use the `ExtractPatchFeatures` transformation instead to extract patch feature maps instead.


## ONNX models
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
