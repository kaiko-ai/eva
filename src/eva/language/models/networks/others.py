"""Models from other providers."""

from eva.language.models import wrappers
from eva.language.models.networks.registry import model_registry


@model_registry.register("others/openai-community-gpt2")
class GPT2(wrappers.HuggingFaceModel):
    """GPT-2 model from OpenAI's community repo."""

    def __init__(self, system_prompt: str | None = None):
        """Initializes model."""
        super().__init__(
            model_name_or_path="openai-community/gpt2", chat_mode=False, system_prompt=system_prompt
        )
