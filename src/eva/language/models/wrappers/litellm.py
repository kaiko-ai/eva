"""LLM wrapper for litellm models."""

from typing import Any, Dict, List

from litellm import batch_completion  # type: ignore
from loguru import logger
from typing_extensions import override

from eva.core.models.wrappers import base


class LiteLLMTextModel(base.BaseModel[List[str], List[str]]):
    """Wrapper class for using litellm for chat-based text generation.

    This wrapper uses litellm's `completion` function which accepts a list of
    message dicts. The `forward` method converts a string prompt into a chat
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

    @override
    def model_forward(self, prompts: List[str]) -> List[str]:
        """Generates text using litellm.

        Args:
            prompts: A list of prompts to be converted into a "user" message.

        Returns:
            A list of generated text responses. Failed generations will contain
            error messages instead of generated text.
        """
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]

        responses = batch_completion(
            model=self._model_name_or_path,
            messages=messages,
            **self._model_kwargs,
        )

        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                error_msg = f"Error generating text for prompt {i}: {response}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            else:
                results.append(response["choices"][0]["message"]["content"])

        return results
