"""LLM wrapper for litellm models."""

from typing import Any, Dict

from litellm import batch_completion
from loguru import logger
from typing_extensions import override

from eva.core.models.wrappers import base


class LiteLLMTextModel(base.BaseModel):
    """Wrapper class for using litellm for chat-based text generation.

    This wrapper uses litellm's `completion` function which accepts a list of
    message dicts. The `generate` method converts a string prompt into a chat
    message with a default "user" role, optionally prepends a system message,
    and includes an API key if provided.
    """

    def __init__(
        self,
        model_name_or_path: str,
        model_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        """Initializes the litellm chat model wrapper.

        Args:
            model_name_or_path: The model identifier (or name) for litellm
                (e.g.,"openai/gpt-4o" or "anthropic/claude-3-sonnet-20240229").
            model_kwargs: Additional keyword arguments to pass during
                generation (e.g., `temperature`, `max_tokens`).
        """
        super().__init__()
        self._model_name_or_path = model_name_or_path
        self._model_kwargs = model_kwargs or {}
        self.load_model()

    @override
    def load_model(self) -> None:
        """Prepares the litellm model.

        Note:
            litellm doesn't require an explicit loading step; models are called
            directly during generation. This method exists for API consistency.
        """
        pass

    def generate(self, prompts: list) -> str:
        """Generates text using litellm.

        Args:
            prompts: A list of prompts to be converted into a "user" message.

        Returns:
            The generated text response.
        """
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        try:
            responses = batch_completion(
                model=self._model_name_or_path, messages=messages, **self._model_kwargs
            )
        except Exception as e:
            logger.error(f"Error generating text: {e}")
        return [response["choices"][0]["message"]["content"] for response in responses]
