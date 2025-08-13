"""LiteLLM vision-language model wrapper."""

from typing import Any, Dict, List

from typing_extensions import override

from eva.language.models import wrappers as language_wrappers
from eva.language.utils.text import messages as language_message_utils
from eva.multimodal.models.typings import TextImageBatch
from eva.multimodal.utils.text import messages as message_utils


class LiteLLMModel(language_wrappers.LiteLLMModel):
    """Wrapper class for LiteLLM vision-language models."""

    @override
    def format_inputs(self, batch: TextImageBatch) -> List[List[Dict[str, Any]]]:
        """Format inputs for LiteLLM processor with byte-encoded images in the prompt."""
        message_batch, image_batch, _, _ = TextImageBatch(*batch)

        message_batch = language_message_utils.batch_insert_system_message(
            message_batch, self.system_message
        )
        message_batch = list(map(language_message_utils.combine_system_messages, message_batch))

        return list(map(message_utils.format_litellm_message, message_batch, image_batch))
