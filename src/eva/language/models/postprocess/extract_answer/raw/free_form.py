"""Postprocessing transforms for extracting answers from raw responses."""

from typing import Dict

from typing_extensions import override

from eva.language.models.postprocess.extract_answer.base import ExtractAnswerFromStructuredOutput


class ExtractAnswerFromRaw(ExtractAnswerFromStructuredOutput):
    """Extracts answers from raw responses and returns structured data."""

    @override
    def _extract_structured_data(self, value: str) -> Dict[str, str]:
        """Returns the raw input string as the answer.

        Args:
            value: The input string.

        Returns:
            A dictionary with format {self.answer_key: "extracted content"}.
        """
        return {self.answer_key: value}
