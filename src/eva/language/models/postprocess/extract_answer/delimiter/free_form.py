"""Postprocessing transforms for extracting free-form answers from delimiter-based responses."""

from typing import Dict

from typing_extensions import override

from eva.language.models.postprocess.extract_answer.base import ExtractAnswerFromStructuredOutput
from eva.language.utils.text import delimiter as delimiter_utils


class ExtractAnswerFromDelimiter(ExtractAnswerFromStructuredOutput):
    """Extracts free-form answers from delimiter-based text responses."""

    def __init__(
        self,
        answer_key: str = "answer",
        case_sensitive: bool = False,
        raise_if_missing: bool = True,
        missing_answer: int = -1,
        missing_limit: int = 5,
        delimiter: str = "####",
    ) -> None:
        """Initialize the transform.

        Args:
            answer_key: The key within the structured object that stores the answer.
            case_sensitive: Whether to treat answers as case sensitive.
            raise_if_missing: Whether to raise an error if an answer is missing.
                If False, will return `missing_answer` instead.
            missing_answer: The integer value to return if the answer is missing
                and `raise_if_missing` is False or the number of missing answers
                are still below `missing_limit`.
            missing_limit: The maximum number of missing responses before raising
                an error, if `raise_if_missing` is True.
            delimiter: The delimiter string to search for (default "####").
        """
        super().__init__(
            answer_key=answer_key,
            case_sensitive=case_sensitive,
            raise_if_missing=raise_if_missing,
            missing_answer=missing_answer,
            missing_limit=missing_limit,
        )
        self.delimiter = delimiter

    @override
    def _extract_structured_data(self, value: str) -> Dict[str, str] | None:
        """Extract delimiter-based data from a string.

        Args:
            value: The input string containing delimiter text.

        Returns:
            Dict[str, str] | None: The extracted delimiter object or None if extraction failed.
        """
        return delimiter_utils.extract_delimiter(
            text=value,
            delimiter=self.delimiter,
            answer_key=self.answer_key,
            case_sensitive=self.case_sensitive,
            answer_options=None,  # No validation for free-form answers
        )
