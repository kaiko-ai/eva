"""Tests for ExtractAnswerFromJson post-processing transform."""

import pytest
import torch

from eva.language.models.postprocess.extract_answer_from_json import ExtractAnswerFromJson


@pytest.fixture
def transform() -> ExtractAnswerFromJson:
    """Return a baseline transform with case-insensitive defaults."""
    return ExtractAnswerFromJson(mapping={"Yes": 1, "No": 0})


def test_call_single_string_returns_tensor(transform: ExtractAnswerFromJson) -> None:
    """A single JSON string should yield an int tensor with trimmed, casefolded lookup."""
    tensor = transform('{"answer": "  YES  "}')

    assert tensor.tolist() == [1]
    assert tensor.dtype == torch.int


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
    transform = ExtractAnswerFromJson(mapping={"Yes": 1}, case_sensitive=True)

    assert transform('{"answer": "Yes"}').tolist() == [1]
    with pytest.raises(ValueError, match="Answer 'yes' not found in mapping"):
        transform('{"answer": "yes"}')


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
    with pytest.raises(ValueError, match="Provided JSON is missing the 'answer' key"):
        transform('{"not_answer": "Yes"}')


def test_init_requires_non_empty_mapping() -> None:
    """An empty mapping should be rejected at construction time."""
    with pytest.raises(ValueError, match="`mapping` must be a non-empty dictionary."):
        ExtractAnswerFromJson(mapping={})
