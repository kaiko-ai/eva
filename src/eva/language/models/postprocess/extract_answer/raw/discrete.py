"""Postprocessing transforms for extracting answers from raw text responses."""

from typing import Dict

from typing_extensions import override

from eva.language.models.postprocess.extract_answer.base import (
    ExtractDiscreteAnswerFromStructuredOutput,
)
from eva.language.utils.text import raw as raw_utils


class ExtractDiscreteAnswerFromRaw(ExtractDiscreteAnswerFromStructuredOutput):
    """Extracts discrete answers from raw text responses and casts them to int tensors."""

    @override
    def _extract_structured_data(self, value: str) -> Dict[str, str] | None:
        """Extract raw data from a string.

        Args:
            value: The input string containing raw text.

        Returns:
            Dict[str, str] | None: The extracted raw object or None if extraction failed.
        """
        return raw_utils.extract_raw(
            text=value,
            answer_options=list(self.mapping.keys()),
            answer_key=self.answer_key,
            case_sensitive=self.case_sensitive,
        )
