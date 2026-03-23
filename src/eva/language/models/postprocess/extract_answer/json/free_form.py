"""Postprocessing transforms for extracting answers from JSON responses."""

from typing import Dict

from typing_extensions import override

from eva.language.models.postprocess.extract_answer.base import ExtractAnswerFromStructuredOutput
from eva.language.utils.text import json as json_utils


class ExtractAnswerFromJson(ExtractAnswerFromStructuredOutput):
    """Extracts answers from JSON responses and returns structured data."""

    @override
    def _extract_structured_data(self, value: str) -> Dict[str, str] | None:
        """Extract JSON data from a string.

        Args:
            value: The input string containing JSON.

        Returns:
            Dict[str, str] | None: The extracted JSON object or None if extraction failed.
        """
        return json_utils.extract_json(value)
