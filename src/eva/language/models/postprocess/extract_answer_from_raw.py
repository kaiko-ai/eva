"""Postprocessing transforms for extracting answers from raw text responses."""

from typing import Dict

from typing_extensions import override

from eva.language.models.postprocess.extract_answer_base import (
    ExtractDiscreteAnswerFromStructuredOutput,
)


class ExtractDiscreteAnswerFromRaw(ExtractDiscreteAnswerFromStructuredOutput):
    """Extracts discrete answers from the end of raw text responses.

    This extractor looks at the last N words of the response and attempts to find
    a valid answer option. It is designed to work with RawMultipleChoicePromptTemplate
    where the model is instructed to provide its final answer at the end of the response.

    Note: All answer options in the mapping must be single words (e.g., "Yes", "No", "A", "B").
    Multi-word answers are not supported.
    """

    def __init__(
        self,
        mapping: Dict[str, int],
        answer_key: str = "answer",
        case_sensitive: bool = False,
        raise_if_missing: bool = True,
        missing_response: int = -1,
        missing_limit: int = 5,
        lookback_words: int = 3,
    ) -> None:
        """Initialize the transform.

        Args:
            mapping: Mapping from answer strings to integer IDs.
            answer_key: Not used for raw extraction (kept for API compatibility).
            case_sensitive: Whether to treat mappings as case sensitive.
            raise_if_missing: Whether to raise an error if an answer is missing
                or not found in the mapping. If False, will return `missing_response`
                instead.
            missing_response: The integer value to return if the answer is missing
                and `raise_if_missing` is False or the number of missing answers
                are still below `missing_limit`.
            missing_limit: The maximum number of missing responses before raising
                an error, if `raise_if_missing` is True.
            lookback_words: Number of words to examine from the end of the response
                when searching for the answer. Default is 10.
        """
        # Validate that all mapping keys are single words
        for key in mapping:
            if not isinstance(key, str) or " " in key.strip():
                raise ValueError(
                    f"All mapping keys must be single words. Found multi-word key: '{key}'"
                )

        super().__init__(
            mapping=mapping,
            answer_key=answer_key,
            case_sensitive=case_sensitive,
            raise_if_missing=raise_if_missing,
            missing_response=missing_response,
            missing_limit=missing_limit,
        )

        if lookback_words < 1:
            raise ValueError("`lookback_words` must be at least 1.")

        self.lookback_words = lookback_words

    @override
    def _extract_structured_data(self, value: str) -> Dict[str, str] | None:
        """Extract answer from the end of a raw text response.

        This method treats the raw text as "structured data" by returning a dictionary
        with the answer key pointing to the extracted answer string, maintaining API
        compatibility with the base class.

        Args:
            value: The raw response string.

        Returns:
            Dict[str, str] | None: Dictionary with the answer key if extraction succeeded,
                None otherwise.
        """
        if not isinstance(value, str) or not value.strip():
            return None

        words = value.strip().split()
        for i in range(1, min(self.lookback_words, len(words)) + 1):
            word = words[-i].strip()
            word = word.strip(".,;:!?\"'()[]{}").strip()
            key = word if self.case_sensitive else word.lower()
            if key in self.mapping:
                return {self.answer_key: word}

        return None
