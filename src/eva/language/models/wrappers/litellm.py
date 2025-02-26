"""LLM wrapper for litellm models."""

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
        model_kwargs: dict | None = None,
    ) -> None:
        """Initializes the litellm chat model wrapper.

        Args:
            model_name_or_path: The model identifier (or name) for litellm (e.g.,
                "openai/gpt-4o" or "anthropic/claude-3-sonnet-20240229").
            model_kwargs: Additional keyword arguments to pass to the model during
                generation (e.g., `temperature`, `max_tokens`, `top_k`, `top_p`).
        """
        super().__init__()
        self._model_name_or_path = model_name_or_path
        self._model_kwargs = model_kwargs if model_kwargs else {}
        self.load_model()

    @override
    def load_model(self) -> None:
        """Prepares the litellm model.

        Note:
            litellm does not require an explicit loading step; models are invoked
            directly during generation. This method exists for API consistency.
        """
        pass

    def generate(self, prompt: str) -> str:
        """Generates text using litellm.

        Args:
            prompt: A string prompt that will be converted into a "user" chat message.

        Returns:
            The generated text response.
        """
        messages = [{"role": "user", "content": prompt}]
        response = completion(
            model=self._model_name_or_path, messages=messages, **self._model_kwargs
        )
        return response["choices"][0]["message"]["content"]
