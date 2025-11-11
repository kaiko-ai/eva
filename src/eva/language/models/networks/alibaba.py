"""Models from Alibaba."""

import torch

from eva.core.utils import requirements
from eva.language.models import wrappers
from eva.language.models.constants import MAX_NEW_TOKENS
from eva.language.models.networks.registry import model_registry


@model_registry.register("alibaba/qwen2-0-5b-instruct")
class Qwen205BInstruct(wrappers.HuggingFaceModel):
    """Qwen2 0.5B Instruct model."""

    def __init__(self, system_prompt: str | None = None, cache_dir: str | None = None):
        """Initialize the model."""
        requirements.check_min_versions(requirements={"torch": "2.5.1", "torchvision": "0.20.1"})
        super().__init__(
            model_name_or_path="Qwen/Qwen2-0.5B-Instruct",
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "cache_dir": cache_dir,
            },
            generation_kwargs={
                "max_new_tokens": MAX_NEW_TOKENS,
            },
            system_prompt=system_prompt,
            chat_mode=True,
        )
