"""Postprocessing transforms for extracting answers from raw responses."""

from typing_extensions import override

from eva.language.models.postprocess.extract_answer.base import ExtractAnswerFromStructuredOutput


class ExtractAnswerFromRaw(ExtractAnswerFromStructuredOutput):
    """Extracts answers from raw responses and returns structured data."""

    @override
    def _extract_structured_data(self, value: str) -> str:
        """Extract raw data from a string; returning the string as is.

        Args:
            value: The input string.

        Returns:
            str: The input string unchanged.
        """
        return value
