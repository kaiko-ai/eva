"""Models from Alibaba."""

import torch

from eva.language.models import wrappers
from eva.language.models.networks.registry import model_registry


@model_registry.register("alibaba/qwen2-0-5b-instruct")
class Qwen205BInstruct(wrappers.HuggingFaceModel):
    """Qwen2 0.5B Instruct model."""

    def __init__(self, system_prompt: str | None = None, cache_dir: str | None = None):
        """Initialize the model."""
        super().__init__(
            model_name_or_path="Qwen/Qwen2-0.5B-Instruct",
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "cache_dir": cache_dir,
            },
            generation_kwargs={
                "max_new_tokens": 512,
            },
            system_prompt=system_prompt,
            chat_mode=True,
        )
