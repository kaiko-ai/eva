# Large-Scale Model Inference with vLLM

This guide shows how to run large-scale vision-language model (VLM) evaluations using *eva*'s vLLM integration. [vLLM](https://github.com/vllm-project/vllm) is a high-throughput inference engine that enables efficient evaluation of large models through features like tensor parallelism and optimized memory management.

## Prerequisites

vLLM is not included in *eva*'s dependencies as it can be complex to install on certain systems (specific CUDA versions, platform compatibility, etc.). You need to install it manually:

```bash
pip install "kaiko-eva[all]"
pip install vllm
```

Refer to the [vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation.html) for platform-specific instructions.

## Quick Start

To evaluate the Qwen2.5-VL 72B model on the PatchCamelyon dataset:

```bash
MODEL_NAME="alibaba/qwen2-5-vl-72b-instruct-vllm" \
VLLM_TENSOR_PARALLEL_SIZE=4 \
ACCELERATOR=cpu \
NUM_DEVICES=1 \
BATCH_SIZE=256 \
eva fit --config configs/multimodal/pathology/online/multiple_choice/patch_camelyon.yaml
```

This command will distribute the model across 4 GPUs (`VLLM_TENSOR_PARALLEL_SIZE`) using vLLM's tensor parallelism.

Note that we set `ACCELERATOR=cpu` and `NUM_DEVICES=1` so that PyTorch Lightning runs on CPU while vLLM handles loading the model across available GPUs. vLLM will automatically use GPUs listed in the `CUDA_VISIBLE_DEVICES` environment variable (should be set automatically by lightning when using `eva`).

We recommend using big batch sizes in the lightning dataloader, since vLLM handles batching internally for optimal throughput.

## How It Works

### The VllmModel Wrapper

*eva* provides a `VllmModel` wrapper class that integrates vLLM for efficient inference. The wrapper handles:

- Automatic chat template formatting
- Image preprocessing for multimodal inputs
- Batched generation with configurable sampling parameters

The `eva.language` and `eva.multimodal` modules have a separate `VllmModel` wrapper implementation, so for LLMs please use `eva.language.models.wrappers.VllmModel` and for VLM evaluation you can use `eva.multimodal.models.wrappers.VllmModel`.
Note that `language` and `multimodal` models have separate model registries, so make sure to register your vLLM model in the correct registry.

## vLLM Settings

The `VllmModel` wrapper uses the following defaults for evaluation:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_model_len` | 32768 | Maximum sequence length |
| `gpu_memory_utilization` | 0.95 | Fraction of GPU memory to use |
| `tensor_parallel_size` | 1 | Number of GPUs for tensor parallelism |
| `temperature` | 0.0 | Sampling temperature (deterministic) |
| `max_tokens` | 8192 | Maximum tokens the model can generate per sample |

To keep in mind:

- Setting `gpu_memory_utilization` below 1.0 helps prevent out-of-memory errors by leaving some GPU memory free.
- `temperature` of 0.0 ensures deterministic outputs during evaluation.

## Custom vLLM Models

To add a new vLLM-enabled model, register it in the model registry:

```python
from eva.multimodal.models import wrappers
from eva.multimodal.models.networks.registry import model_registry

@model_registry.register("your-org/your-model-vllm")
class YourModelVllm(wrappers.VllmModel):
    """Your custom vLLM model."""

    def __init__(self, system_prompt: str | None = None):
        super().__init__(
            model_name_or_path="your-org/your-model",
            model_kwargs={
                "tensor_parallel_size": 2,  # Adjust based on model size
            },
            image_position="before_text",
            system_prompt=system_prompt,
        )
```

After doing so, you can use this model by setting `MODEL_NAME="your-org/your-model-vllm"` in your evaluation command. Just make sure that you import your new model / or the module where your model class is defined somewhere in a `__init__.py` or in your evaluation script such that the registration gets triggered.

To set vLLM-specific parameters, pass them via the `model_kwargs` dictionary.
To solve OOM issues, tuning `max_num_seqs` and `dtype` can help, depending on the model you want to load and the given hardware setup. 

If you want to add an LLM, please use the corresponding `VllmModel` and `model_registry` from `eva.language` instead of `eva.multimodal`.

Note that not all Huggingface compatible models are out-of-the box compatible with vLLM. Please refer to the [vLLM documentation](https://docs.vllm.ai/en/latest/) for a list of supported models. For custom models, you might need to implement your own vLLM compatible model class and registering it with vLLMs model registry (`vllm.ModelRegistry.register_model`) before using it with *eva*.

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:

1. Adjust parameters such as `tensor_parallel_size`, `max_model_len`, `gpu_memory_utilization`, `max_num_batched_tokens`. Those parameters you can set via `model_kwargs` when initializing the `VllmModel` wrapper.
3. Make sure you use a recent vLLM version and matching torch & CUDA versions. Also enable V1 engine if you are using a vLLM version where this is not the default yet by setting `VLLM_USE_V1="1"` (this offers range of significant performance improvements to V0 engine).
4. Monitor GPU memory usage (GPU memory utilization should be equally distributed across all used GPUs and below `gpu_memory_utilization`). If GPU memory only rises on a single GPU, that might indicate an issue with the tensor parallelism setup or vLLM not detecting all GPUs correctly.

Reducing the dataloader batch size usually doesn't help, as vLLM handles batching internally. 

### Model Loading Issues

Ensure you have:

- Sufficient disk space for model weights
- Correct HuggingFace access tokens for gated models: `HF_TOKEN=<your-token>`
