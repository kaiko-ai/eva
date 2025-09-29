"""LiteLLM vision-language model wrapper."""

import logging
from typing import Any, Dict, List

from typing_extensions import override

from eva.language.models import wrappers as language_wrappers
from eva.language.models.typings import ModelOutput
from eva.language.utils.text import messages as language_message_utils
from eva.multimodal.models.typings import TextImageBatch
from eva.multimodal.models.wrappers import base
from eva.multimodal.utils.batch import unpack_batch
from eva.multimodal.utils.text import messages as message_utils


class LiteLLMModel(base.VisionLanguageModel):
    """Wrapper class for LiteLLM vision-language models."""

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

        self.language_model = language_wrappers.LiteLLMModel(
            model_name=model_name,
            model_kwargs=model_kwargs,
            system_prompt=system_prompt,
            log_level=log_level,
        )

    @override
    def format_inputs(self, batch: TextImageBatch) -> List[List[Dict[str, Any]]]:
        message_batch, image_batch, _, _ = unpack_batch(batch)

        message_batch = language_message_utils.batch_insert_system_message(
            message_batch, self.system_message
        )
        message_batch = list(map(language_message_utils.combine_system_messages, message_batch))

        return list(map(message_utils.format_litellm_message, message_batch, image_batch))

    @override
    def model_forward(self, batch: List[List[Dict[str, Any]]]) -> ModelOutput:
        return self.language_model.model_forward(batch)
