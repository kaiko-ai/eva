"""Tests for ExtractDiscreteAnswerFromJson post-processing transform."""

import re

import pytest
import torch

from eva.language.models.postprocess.extract_answer.json.discrete import (
    ExtractDiscreteAnswerFromJson,
)


@pytest.fixture
def transform() -> ExtractDiscreteAnswerFromJson:
    """Return a baseline transform with case-insensitive defaults."""
    return ExtractDiscreteAnswerFromJson(mapping={"Yes": 1, "No": 0}, missing_limit=0)


def test_call_single_string_returns_tensor(transform: ExtractDiscreteAnswerFromJson) -> None:
    """A single JSON string should yield an int tensor with trimmed, casefolded lookup."""
    result = transform('{"answer": "  YES  "}')

    assert result.tolist() == [1]
    assert result.dtype == torch.long


def test_call_list_parses_code_fences(transform: ExtractDiscreteAnswerFromJson) -> None:
    """Lists of responses should parse markdown fenced JSON and preserve order."""
    result = transform(['```json\n{"answer": "No"}\n```', '{"answer": "Yes"}'])

    assert result.tolist() == [0, 1]


def test_call_ignores_surrounding_text(transform: ExtractDiscreteAnswerFromJson) -> None:
    """Noise around the JSON blob should be ignored by extract_json."""
    raw_response = "Final thoughts:\n" "```json\n" '{"answer": "Yes"}\n' "```\n" "Thank you!"
    result = transform(raw_response)

    assert result.tolist() == [1]


def test_custom_answer_key_respected() -> None:
    """Custom answer_key should be used when extracting responses."""
    transform = ExtractDiscreteAnswerFromJson(mapping={"blue": 2}, answer_key="choice")
    result = transform('{"choice": "Blue"}')

    assert result.tolist() == [2]


def test_case_sensitive_behavior() -> None:
    """Case-sensitive mode should only match exact variants."""
    transform = ExtractDiscreteAnswerFromJson(
        mapping={"yes": 1}, case_sensitive=True, missing_limit=0
    )

    result = transform('{"answer": "yes"}')
    assert result.tolist() == [1]
    with pytest.raises(ValueError, match=re.escape("Answer 'Yes' not found in mapping: ['yes']")):
        transform('{"answer": "Yes"}')


def test_missing_answer_maps_to_fallback_when_allowed() -> None:
    """Missing answers should return the configured fallback when raising is disabled."""
    transform = ExtractDiscreteAnswerFromJson(
        mapping={"yes": 1},
        raise_if_missing=False,
        missing_answer=-42,
    )

    result = transform('{"answer": "maybe"}')

    assert result.tolist() == [-42]


def test_missing_answer_key_raises(transform: ExtractDiscreteAnswerFromJson) -> None:
    """Responses without the answer key should raise a descriptive error."""
    with pytest.raises(ValueError, match="Found 1 responses without valid structured data"):
        transform('{"not_answer": "Yes"}')


def test_missing_limit_raises_after_threshold() -> None:
    """Missing JSON responses should respect the configured missing_limit."""
    transform = ExtractDiscreteAnswerFromJson(
        mapping={"no": 0, "yes": 1},
        missing_limit=3,
        missing_answer=-99,
    )
    result1 = transform("unknown")
    assert result1.tolist() == [-99]
    result2 = transform(["unknown", "unknown"])
    assert result2.tolist() == [-99, -99]
    with pytest.raises(ValueError, match="Found 4 responses without valid structured data."):
        transform("unknown")


def test_init_requires_non_empty_mapping() -> None:
    """An empty mapping should be rejected at construction time."""
    with pytest.raises(ValueError, match="`mapping` must be a non-empty dictionary."):
        ExtractDiscreteAnswerFromJson(mapping={})
