import logging
import os
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

from eva.multimodal.models.typings import ModelType, TextBatch, TextImageBatch, VisionLanguageOutput
from eva.multimodal.models.utils.batch import unpack_batch
from eva.multimodal.models.wrappers import base
from eva.multimodal.utils.text import messages as message_utils

LITELLM_LOG_LEVEL_TO_WARN = os.environ.get("LITELLM_LOG_LEVEL_TO_WARN", True)
RETRYABLE_ERRORS = (
    RateLimitError,
    Timeout,
    InternalServerError,
    APIConnectionError,
    ServiceUnavailableError,
)


class LiteLLMModel(base.VisionLanguageModel):
    """Lightweight wrapper for LiteLLM models."""

    model_type: ModelType = "api"
    """API model type."""

    def __init__(
        self,
        model_name: str,
        api_kwargs: Dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ):
        """Initialize the LiteLLM Wrapper.

        Args:
            model_name: The name of the model to use.
            api_kwargs: Additional API arguments to pass to the API such as API version for Azure.
            system_prompt: The system prompt to use (optional).
        """
        super().__init__(system_prompt=system_prompt)

        self.model_name = model_name
        litellm.suppress_debug_info = True
        self.max_tokens = None

        if api_kwargs:
            self.max_tokens = api_kwargs.pop("max_tokens", None)
            for key, value in api_kwargs.items():
                os.environ[key] = value

        if LITELLM_LOG_LEVEL_TO_WARN:
            logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    @override
    def format_inputs(self, batch: TextImageBatch | TextBatch) -> List[List[Dict[str, Any]]]:
        """Format inputs for LiteLLM processor with byte-encoded images in the prompt."""
        message_batch, image_batch, _, _ = unpack_batch(batch)

        message_batch = message_utils.batch_insert_system_message(
            message_batch, self.system_message
        )
        message_batch = list(map(message_utils.combine_system_messages, message_batch))

        return list(map(message_utils.format_litellm_message, message_batch, image_batch))

    @backoff.on_exception(
        backoff.expo,
        RETRYABLE_ERRORS,
        max_tries=20,
        jitter=backoff.full_jitter,
        on_backoff=lambda details: logger.warning(
            f"Retrying due to {details.get('exception') or 'Unknown error'}"
        ),
    )
    def model_forward(self, batch: List[List[Dict[str, Any]]]) -> VisionLanguageOutput:
        """API model calls for text generation."""
        outputs = batch_completion(
            model=self.model_name, messages=batch, max_tokens=self.max_tokens
        )
        self._raise_exceptions(outputs)

        ins = [message_utils.messages_to_string(messages) for messages in batch]
        outs = [
            output["choices"][0]["message"]["content"]
            for output in outputs
            if output["choices"][0]["message"]["role"] == "assistant"
        ]

        return {"processed_input": ins, "output": outs}

    def _raise_exceptions(self, outputs: list):
        for output in outputs:
            if isinstance(output, Exception):
                logger.error(f"Model {self.model_name} encountered an error: {output}")
                raise output
