"""Postprocessing transforms for extracting answers from boxed responses."""

from typing import Dict

from typing_extensions import override

from eva.language.models.postprocess.extract_answer.base import (
    ExtractDiscreteAnswerFromStructuredOutput,
)
from eva.language.utils.text import boxed as boxed_utils


class ExtractDiscreteAnswerFromBoxed(ExtractDiscreteAnswerFromStructuredOutput):
    """Extracts discrete answers from boxed responses and casts them to int tensors."""

    @override
    def _extract_structured_data(self, value: str) -> Dict[str, str] | None:
        r"""Extract boxed data from a string.

        Args:
            value: The input string containing \\boxed{}.

        Returns:
            Dict[str, str] | None: The extracted boxed content or None if extraction failed.
        """
        return boxed_utils.extract_boxed(value)
