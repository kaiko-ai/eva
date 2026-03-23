"""Postprocessing transforms for extracting answers from boxed responses."""

from typing import Dict

from typing_extensions import override

from eva.language.models.postprocess.extract_answer.base import ExtractAnswerFromStructuredOutput
from eva.language.utils.text import boxed as boxed_utils


class ExtractAnswerFromBoxed(ExtractAnswerFromStructuredOutput):
    """Extracts answers from boxed responses and returns structured data."""

    @override
    def _extract_structured_data(self, value: str) -> Dict[str, str] | None:
        r"""Extract boxed data from a string.

        Args:
            value: The input string containing \\boxed{}.

        Returns:
            A dictionary with format {self.answer_key: "extracted content"} or
                None if extraction failed.
        """
        answer = boxed_utils.extract_boxed(value)
        return {self.answer_key: answer} if answer is not None else None
