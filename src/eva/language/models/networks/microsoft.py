"""Models from Microsoft."""

from eva.language.models import wrappers
from eva.language.models.constants import MAX_NEW_TOKENS
from eva.language.models.networks.registry import model_registry


@model_registry.register("microsoft/phi-3-mini-4k-instruct")
class Phi3Mini4KInstruct(wrappers.HuggingFaceModel):
    """Phi 3 Mini 4K Instruct model."""

    def __init__(self, system_prompt: str | None = None):
        """Initialize the model."""
        super().__init__(
            model_name_or_path="microsoft/Phi-3-mini-4k-instruct",
            model_class="AutoModelForCausalLM",
            model_kwargs={
                "torch_dtype": "auto",
            },
            generation_kwargs={
                "max_new_tokens": MAX_NEW_TOKENS,
            },
            system_prompt=system_prompt,
        )
