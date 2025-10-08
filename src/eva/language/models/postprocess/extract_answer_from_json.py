"""Postprocessing transforms for extracting answers from JSON responses."""

from typing import Dict, List, Union

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
    ) -> None:
        """Initialize the transform.

        Args:
            mapping: Mapping from answer strings to integer IDs.
            answer_key: The key within the JSON object that stores the answer.
            case_sensitive: Whether to treat mappings as case sensitive.
            raise_if_missing: Whether to raise an error if the answer key is missing
                or if the extracted answer is not in the mapping. If False, will
                return `missing_response` instead.
            missing_response: The integer value to return if the answer is missing
                and `raise_if_missing` is False.
        """
        if not mapping:
            raise ValueError("`mapping` must be a non-empty dictionary.")

        self.answer_key = answer_key
        self.case_sensitive = case_sensitive
        self.raise_if_missing = raise_if_missing
        self.missing_response = missing_response

        self.mapping = {k if case_sensitive else k.lower(): v for k, v in mapping.items()}

    def __call__(self, values: Union[str, List[str]]) -> torch.Tensor:
        """Convert JSON string(s) to a tensor of integer labels."""
        if not isinstance(values, (list, tuple)):
            values = [values]

        jsons = list(map(json_utils.extract_json, values))
        answers = list(map(self._extract_answer, jsons))

        return torch.tensor(answers, dtype=torch.int)

    def _extract_answer(self, json_obj: Dict[str, str]) -> int:
        if self.answer_key not in json_obj:
            raise ValueError(f"Provided JSON is missing the '{self.answer_key}' key")
        answer = json_obj[self.answer_key].strip()

        if not self.case_sensitive:
            answer = answer.lower()

        if answer not in self.mapping:
            if self.raise_if_missing:
                raise ValueError(
                    f"Answer '{answer}' not found in mapping: {list(self.mapping.keys())}"
                )
            else:
                logger.warning(
                    f"Answer '{answer}' not found in mapping, "
                    f"returning {self.missing_response} instead."
                )
                return self.missing_response

        return self.mapping[answer]
