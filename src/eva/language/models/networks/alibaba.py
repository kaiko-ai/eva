"""Models from Alibaba."""

import os

import torch

from eva.core.utils import requirements
from eva.language.models import wrappers
from eva.language.models.constants import MAX_NEW_TOKENS
from eva.language.models.networks.registry import model_registry
from eva.language.utils import imports as import_utils


@model_registry.register("alibaba/qwen2-0-5b-instruct")
class Qwen205BInstruct(wrappers.HuggingFaceModel):
    """Qwen2 0.5B Instruct model."""

    def __init__(self, system_prompt: str | None = None):
        """Initialize the model."""
        requirements.check_min_versions(requirements={"torch": "2.5.1", "torchvision": "0.20.1"})
        super().__init__(
            model_name_or_path="Qwen/Qwen2-0.5B-Instruct",
            model_class="AutoModelForCausalLM",
            model_kwargs={
                "torch_dtype": torch.bfloat16,
            },
            generation_kwargs={
                "max_new_tokens": MAX_NEW_TOKENS,
            },
            system_prompt=system_prompt,
        )


@model_registry.register("alibaba/qwen3-8b")
class Qwen3_8B(wrappers.HuggingFaceModel):
    """Qwen3 8B model."""

    def __init__(self, system_prompt: str | None = None):
        """Initialize the model."""
        requirements.check_min_versions(requirements={"torch": "2.5.1", "torchvision": "0.20.1"})
        super().__init__(
            model_name_or_path="Qwen/Qwen3-8B",
            model_class="AutoModelForCausalLM",
            model_kwargs={
                "torch_dtype": "auto",
            },
            generation_kwargs={
                "max_new_tokens": MAX_NEW_TOKENS,
            },
            system_prompt=system_prompt,
        )


if import_utils.is_vllm_available():

    @model_registry.register("alibaba/qwen2-5-72b-instruct-vllm")
    class Qwen2572BInstructVllm(wrappers.VllmModel):
        """Qwen2.5 72B Instruct model."""

        def __init__(
            self,
            system_prompt: str | None = None,
        ):
            """Initialize the model."""
            super().__init__(
                model_name_or_path="Qwen/Qwen2.5-72B-Instruct",
                model_kwargs={
                    "tensor_parallel_size": int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", 4)),
                },
                system_prompt=system_prompt,
            )
