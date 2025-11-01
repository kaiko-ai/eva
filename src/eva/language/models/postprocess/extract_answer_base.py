"""Base classes for postprocessing transforms."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import torch
from loguru import logger


class ExtractDiscreteAnswerFromStructuredOutput(ABC):
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

        self.answer_key = answer_key
        self.case_sensitive = case_sensitive
        self.raise_if_missing = raise_if_missing
        self.missing_answer = missing_answer
        self.missing_limit = missing_limit

        self.missing_count = 0
        self.mapping = {k if case_sensitive else k.lower(): v for k, v in mapping.items()}

    @abstractmethod
    def _extract_structured_data(self, value: str) -> Dict[str, str] | None:
        """Extract structured data from a string.

        Args:
            value: The input string containing structured data.

        Returns:
            Dict[str, str] | None: The extracted structured data as a dictionary
                or None if extraction failed.
        """
        pass

    def __call__(self, values: Union[str, List[str]]) -> torch.Tensor:
        """Convert structured string(s) to a tensor of integer labels."""
        if not isinstance(values, (list, tuple)):
            values = [values]

        structured_data = list(map(self._extract_structured_data, values))
        answers = list(map(self._extract_answer, structured_data))

        return torch.tensor(answers, dtype=torch.long)

    def _extract_answer(self, structured_obj: Dict[str, str] | None) -> int:
        if structured_obj is None or self.answer_key not in structured_obj:
            self.missing_count += 1
            if self.raise_if_missing and self.missing_count > self.missing_limit:
                raise ValueError(
                    f"Found {self.missing_count} responses without valid structured data."
                )
            else:
                logger.warning(
                    f"Failed to extract answer from response: {structured_obj}, "
                    f"returning {self.missing_answer} instead."
                )
                return self.missing_answer

        return self._apply_mapping(structured_obj[self.answer_key])

    def _apply_mapping(self, value: Any) -> int:
        key = value if self.case_sensitive else str(value).strip().lower()
        if key not in self.mapping:
            if self.raise_if_missing and self.missing_count >= self.missing_limit:
                raise ValueError(
                    f"Answer '{key}' not found in mapping: {list(self.mapping.keys())}"
                )
            else:
                logger.warning(
                    f"Answer '{key}' not found in mapping, "
                    f"returning {self.missing_answer} instead."
                )
                self.missing_count += 1
                return self.missing_answer
        return self.mapping[key]
