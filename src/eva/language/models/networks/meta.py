"""Models from Microsoft."""

import os

from eva.language.models import wrappers
from eva.language.models.constants import MAX_NEW_TOKENS
from eva.language.models.networks.registry import model_registry


@model_registry.register("meta/llama-3.2-3b-instruct")
class Lllama3_2_3BInstruct(wrappers.HuggingFaceModel):
    """Meta Llama 3.2 3B Instruct model."""

    def __init__(self, system_prompt: str | None = None):
        """Initialize the model."""
        if not os.getenv("HF_TOKEN"):
            raise ValueError("HF_TOKEN env variable must be set.")

        super().__init__(
            model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
            model_class="AutoModelForCausalLM",
            model_kwargs={
                "torch_dtype": "auto",
            },
            generation_kwargs={
                "max_new_tokens": MAX_NEW_TOKENS,
            },
            system_prompt=system_prompt,
        )
