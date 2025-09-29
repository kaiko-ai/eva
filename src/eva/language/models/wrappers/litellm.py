"""LiteLLM language model wrapper."""

import logging
from typing import Any, Dict, List

import backoff
import litellm
from litellm import batch_completion
from litellm.exceptions import (
    APIConnectionError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from loguru import logger
from typing_extensions import override

from eva.language.models.typings import ModelOutput, TextBatch
from eva.language.models.wrappers import base
from eva.language.utils.text import messages as message_utils

RETRYABLE_ERRORS = (
    RateLimitError,
    Timeout,
    InternalServerError,
    APIConnectionError,
    ServiceUnavailableError,
)


class LiteLLMModel(base.LanguageModel):
    """Wrapper class for LiteLLM language models."""

    _default_model_kwargs = {
        "temperature": 0.0,
        "max_completion_tokens": 1024,
        "top_p": 1.0,
        "seed": 42,
    }
    """Default API model parameters for evaluation."""

    def __init__(
        self,
        model_name: str,
        model_kwargs: Dict[str, Any] | None = None,
        system_prompt: str | None = None,
        log_level: int | None = logging.INFO,
    ):
        """Initialize the LiteLLM Wrapper.

        Args:
            model_name: The name of the model to use.
            model_kwargs: Additional keyword arguments to pass during
                generation (e.g., `temperature`, `max_tokens`).
            system_prompt: The system prompt to use (optional).
            log_level: Optional logging level for LiteLLM. Defaults to WARNING.
        """
        super().__init__(system_prompt=system_prompt)

        self.model_name = model_name
        self.model_kwargs = self._default_model_kwargs | (model_kwargs or {})

        litellm.suppress_debug_info = True
        litellm.drop_params = True

        if log_level is not None:
            logging.getLogger("LiteLLM").setLevel(log_level)

    @override
    def format_inputs(self, batch: TextBatch) -> List[List[Dict[str, Any]]]:
        """Formats inputs for LiteLLM.

        Args:
            batch: A batch of text inputs.

        Returns:
            A list of messages in the following format:
            [
                {
                    "role": ...
                    "content": ...
                },
                ...
            ]
        """
        message_batch, _, _ = TextBatch(*batch)

        message_batch = message_utils.batch_insert_system_message(
            message_batch, self.system_message
        )
        message_batch = list(map(message_utils.combine_system_messages, message_batch))

        return list(map(message_utils.format_chat_message, message_batch))

    @override
    @backoff.on_exception(
        backoff.expo,
        RETRYABLE_ERRORS,
        max_tries=20,
        jitter=backoff.full_jitter,
        on_backoff=lambda details: logger.warning(
            f"Retrying due to {details.get('exception') or 'Unknown error'}"
        ),
    )
    def model_forward(self, batch: List[List[Dict[str, Any]]]) -> ModelOutput:
        """Generates output text through API calls via LiteLLM's batch completion functionality."""
        outputs = batch_completion(model=self.model_name, messages=batch, **self.model_kwargs)
        self._raise_exceptions(outputs)

        generated_text = [
            output["choices"][0]["message"]["content"]
            for output in outputs
            if output["choices"][0]["message"]["role"] == "assistant"
        ]
        return ModelOutput(generated_text=generated_text)

    def _raise_exceptions(self, outputs: list):
        for output in outputs:
            if isinstance(output, Exception):
                logger.error(f"Model {self.model_name} encountered an error: {output}")
                raise output
