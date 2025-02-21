"""LLM wrapper for litellm models."""

from typing import Any, Dict, Optional

from litellm import completion
from typing_extensions import override

from eva.core.models.wrappers import base


class LiteLLMTextModel(base.BaseModel):
    """Wrapper class for using litellm for chat-based text generation.

    This wrapper uses litellm's `completion` function which accepts a list of
    message dictionaries. The `generate` method converts a string prompt into a chat
    message with a default role of "user", optionally prepends a system message, and
    includes an API key if provided.
    """

    def __init__(
        self,
        model_name_or_path: str,
    ) -> None:
        """Initializes the litellm chat model wrapper.

        Args:
            model_name_or_path: The model identifier (or name) for litellm (e.g.,
                "openai/gpt-4o" or "anthropic/claude-3-sonnet-20240229").
        """
        super().__init__()
        self._model_name_or_path = model_name_or_path
        self.load_model()

    @override
    def load_model(self) -> None:
        """Prepares the litellm model.

        Note:
            litellm does not require an explicit loading step; models are invoked
            directly during generation. This method exists for API consistency.
        """
        pass

    def generate(self, prompt: str, **generate_kwargs) -> str:
        """Generates text using litellm.

        Args:
            prompt: A string prompt that will be converted into a "user" chat message.
            generate_kwargs: Additional parameters for generation (e.g., max_tokens).

        Returns:
            The generated text response.
        """
        messages = [{"role": "user", "content": prompt}]
        response = completion(model=self._model_name_or_path, messages=messages, **generate_kwargs)
        return response["choices"][0]["message"]["content"]
