"""Tests for ExtractAnswerFromJson post-processing transform."""

import re

import pytest
import torch

from eva.language.models.postprocess.extract_answer_from_json import ExtractAnswerFromJson


@pytest.fixture
def transform() -> ExtractAnswerFromJson:
    """Return a baseline transform with case-insensitive defaults."""
    return ExtractAnswerFromJson(mapping={"Yes": 1, "No": 0}, missing_limit=0)


def test_call_single_string_returns_tensor(transform: ExtractAnswerFromJson) -> None:
    """A single JSON string should yield an int tensor with trimmed, casefolded lookup."""
    tensor = transform('{"answer": "  YES  "}')

    assert tensor.tolist() == [1]
    assert tensor.dtype == torch.long


def test_call_list_parses_code_fences(transform: ExtractAnswerFromJson) -> None:
    """Lists of responses should parse markdown fenced JSON and preserve order."""
    tensor = transform(['```json\n{"answer": "No"}\n```', '{"answer": "Yes"}'])

    assert tensor.tolist() == [0, 1]


def test_call_ignores_surrounding_text(transform: ExtractAnswerFromJson) -> None:
    """Noise around the JSON blob should be ignored by extract_json."""
    raw_response = "Final thoughts:\n" "```json\n" '{"answer": "Yes"}\n' "```\n" "Thank you!"
    tensor = transform(raw_response)

    assert tensor.tolist() == [1]


def test_custom_answer_key_respected() -> None:
    """Custom answer_key should be used when extracting responses."""
    transform = ExtractAnswerFromJson(mapping={"blue": 2}, answer_key="choice")
    tensor = transform('{"choice": "Blue"}')

    assert tensor.tolist() == [2]


def test_case_sensitive_behavior() -> None:
    """Case-sensitive mode should only match exact variants."""
    transform = ExtractAnswerFromJson(mapping={"yes": 1}, case_sensitive=True, missing_limit=0)

    assert transform('{"answer": "yes"}').tolist() == [1]
    with pytest.raises(ValueError, match=re.escape("Answer 'Yes' not found in mapping: ['yes']")):
        transform('{"answer": "Yes"}')


def test_missing_answer_maps_to_fallback_when_allowed() -> None:
    """Missing answers should return the configured fallback when raising is disabled."""
    transform = ExtractAnswerFromJson(
        mapping={"yes": 1},
        raise_if_missing=False,
        missing_response=-42,
    )

    tensor = transform('{"answer": "maybe"}')

    assert tensor.tolist() == [-42]


def test_missing_answer_key_raises(transform: ExtractAnswerFromJson) -> None:
    """Responses without the answer key should raise a descriptive error."""
    with pytest.raises(ValueError, match="Found 1 responses without JSON objects"):
        transform('{"not_answer": "Yes"}')


def test_missing_limit_raises_after_threshold() -> None:
    """Missing JSON responses should respect the configured missing_limit."""
    transform = ExtractAnswerFromJson(
        mapping={"no": 0, "yes": 1},
        missing_limit=3,
        missing_response=-99,
    )
    assert transform("unknown").tolist() == [-99]
    assert transform(["unknown", "unknown"]).tolist() == [-99, -99]
    with pytest.raises(ValueError, match="Found 4 responses without JSON objects."):
        transform("unknown")


def test_init_requires_non_empty_mapping() -> None:
    """An empty mapping should be rejected at construction time."""
    with pytest.raises(ValueError, match="`mapping` must be a non-empty dictionary."):
        ExtractAnswerFromJson(mapping={})
