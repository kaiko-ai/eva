"""Base classes for postprocessing transforms."""

from abc import ABC, abstractmethod
from typing import Dict, List, Union

import torch
from loguru import logger
from typing_extensions import override


class ExtractAnswerFromStructuredOutput(ABC):
    """Base class for extracting answers from structured output formats."""

    def __init__(
        self,
        answer_key: str = "answer",
        case_sensitive: bool = False,
        raise_if_missing: bool = True,
        missing_answer: str | None = None,
        missing_limit: int = 5,
        return_dict: bool = False,
    ) -> None:
        """Initialize the transform.

        Args:
            answer_key: The key/tag within the structured object that stores the answer.
            case_sensitive: Whether to treat mappings as case sensitive.
            raise_if_missing: Whether to raise an error if an answer is missing
                or not found in the mapping. If False, will return `missing_answer`
                instead.
            missing_answer: The integer value to return if the answer is missing
                and `raise_if_missing` is False or the number of missing answers
                are still below `missing_limit`.
            missing_limit: The maximum number of missing responses before raising
                an error, if `raise_if_missing` is True.
            return_dict: Whether to return the extracted structured data as a dictionary.
                If False, only the answer string under "answer_key" will be returned.
                Enabling this can be handy if additional fields are extracted alongside
                the answer and are needed downstream e.g. for logging or metrics calculation.
                The logic furthermore ensures that the returned dictionary always contains
                an "answer_key" entry.
        """
        self.answer_key = answer_key
        self.case_sensitive = case_sensitive
        self.raise_if_missing = raise_if_missing
        self.missing_answer = missing_answer
        self.return_dict = return_dict
        self.missing_limit = int(missing_limit)
        self.missing_count = 0

    @abstractmethod
    def _extract_structured_data(self, value: str) -> Dict[str, str] | None:
        """Extract structured data from a string.

        Args:
            value: The input string containing structured data.

        Returns:
            Dict[str, str] | None: The extracted structured data as a dictionary
                or None if extraction failed.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _extract_answer(self, value: str) -> str | Dict[str, str] | None:
        """Extracts the answer from the input text.

        If extraction fails, handles missing data according, either returns
        `self.missing_answer` or raises an error, based on configuration.

        Args:
            value: A string to extract the answer from.

        Returns:
            str | None: The extracted answer string, or None if extraction failed.
        """
        structured_data = self._extract_structured_data(value)

        if structured_data is None or self.answer_key not in structured_data:
            self.missing_count += 1
            if self.raise_if_missing and self.missing_count > self.missing_limit:
                raise ValueError(
                    f"Found {self.missing_count} responses without valid structured data."
                )
            logger.warning(
                f"Failed to extract answer from response: {structured_data}, "
                f"returning {self.missing_answer} instead."
            )
            return self.missing_answer

        return structured_data if self.return_dict else str(structured_data[self.answer_key])

    def __call__(self, values: Union[str, List[str]]) -> List[str | Dict[str, str] | None]:
        """Extracts answers from text(s)."""
        if not isinstance(values, (list, tuple)):
            values = [values]

        return list(map(self._extract_answer, values))


class ExtractDiscreteAnswerFromStructuredOutput(ExtractAnswerFromStructuredOutput, ABC):
    """Base class for extracting discrete answers from structured output formats."""

    def __init__(
        self,
        mapping: Dict[str, int],
        answer_key: str = "answer",
        case_sensitive: bool = False,
        raise_if_missing: bool = True,
        missing_answer: int = -1,
        missing_limit: int = 5,
    ) -> None:
        """Initialize the transform.

        Args:
            mapping: Mapping from answer strings to integer IDs.
            answer_key: The key/tag within the structured object that stores the answer.
            case_sensitive: Whether to treat mappings as case sensitive.
            raise_if_missing: Whether to raise an error if an answer is missing
                or not found in the mapping. If False, will return `missing_answer`
                instead.
            missing_answer: The integer value to return if the answer is missing
                and `raise_if_missing` is False or the number of missing answers
                are still below `missing_limit`.
            missing_limit: The maximum number of missing responses before raising
                an error, if `raise_if_missing` is True.
        """
        if not isinstance(mapping, dict) or len(mapping) == 0:
            raise ValueError("`mapping` must be a non-empty dictionary.")

        super().__init__(
            answer_key=answer_key,
            case_sensitive=case_sensitive,
            raise_if_missing=raise_if_missing,
            missing_limit=missing_limit,
            return_dict=False,
        )

        self.missing_discrete_answer = int(missing_answer)
        self.mapping = {k if case_sensitive else k.lower(): v for k, v in mapping.items()}

    @override
    def __call__(self, values: Union[str, List[str]]) -> torch.Tensor:
        """Extracts answers from text(s) and maps them to discrete integers."""
        if not isinstance(values, (list, tuple)):
            values = [values]

        answers = list(map(self._extract_answer, values))
        discrete_answers = [self._apply_mapping(a) for a in answers]  # type: ignore

        return torch.tensor(discrete_answers, dtype=torch.long)

    def _apply_mapping(self, value: str | None) -> int:
        if value is None or value == self.missing_answer:
            return self.missing_discrete_answer

        key = value if self.case_sensitive else str(value).strip().lower()
        if key not in self.mapping:
            if self.raise_if_missing and self.missing_count >= self.missing_limit:
                raise ValueError(
                    f"Answer '{key}' not found in mapping: {list(self.mapping.keys())}"
                )
            logger.warning(
                f"Answer '{key}' not found in mapping, "
                f"returning {self.missing_discrete_answer} instead."
            )
            self.missing_count += 1
            return self.missing_discrete_answer
        return self.mapping[key]
