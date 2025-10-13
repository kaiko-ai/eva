"""Postprocessing transforms for extracting answers from JSON responses."""

from typing import Any, Dict, List, Union

import torch
from loguru import logger

from eva.language.utils.text import json as json_utils


class ExtractAnswerFromJson:
    """Extracts discrete answers from JSON responses and casts them to int tensors."""

    def __init__(
        self,
        mapping: Dict[str, int],
        answer_key: str = "answer",
        case_sensitive: bool = False,
        raise_if_missing: bool = True,
        missing_response: int = -1,
        missing_limit: int = 5,
    ) -> None:
        """Initialize the transform.

        Args:
            mapping: Mapping from answer strings to integer IDs.
            answer_key: The key within the JSON object that stores the answer.
            case_sensitive: Whether to treat mappings as case sensitive.
            raise_if_missing: Whether to raise an error if an answer is missing
                or not found in the mapping. If False, will return `missing_response`
                instead.
            missing_response: The integer value to return if the answer is missing
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
        self.missing_response = missing_response
        self.missing_limit = missing_limit

        self.missing_count = 0
        self.mapping = {k if case_sensitive else k.lower(): v for k, v in mapping.items()}

    def __call__(self, values: Union[str, List[str]]) -> torch.Tensor:
        """Convert JSON string(s) to a tensor of integer labels."""
        if not isinstance(values, (list, tuple)):
            values = [values]

        jsons = list(map(json_utils.extract_json, values))
        answers = list(map(self._extract_answer, jsons))  # type: ignore

        return torch.tensor(answers, dtype=torch.long)

    def _extract_answer(self, json_obj: Dict[str, str] | None) -> int:
        if json_obj is None or self.answer_key not in json_obj:
            self.missing_count += 1
            if self.raise_if_missing and self.missing_count > self.missing_limit:
                raise ValueError(f"Found {self.missing_count} responses without JSON objects.")
            else:
                logger.warning(
                    f"Failed to extract answer from response: {json_obj}, "
                    f"returning {self.missing_response} instead."
                )
                return self.missing_response

        return self._apply_mapping(json_obj[self.answer_key])

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
                    f"returning {self.missing_response} instead."
                )
                self.missing_count += 1
                return self.missing_response
        return self.mapping[key]
