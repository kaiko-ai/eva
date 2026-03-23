"""Tests for ExtractDiscreteAnswerFromBoxed post-processing transform."""

import re

import pytest
import torch

from eva.language.models.postprocess.extract_answer.boxed.discrete import (
    ExtractDiscreteAnswerFromBoxed,
)


@pytest.fixture
def transform() -> ExtractDiscreteAnswerFromBoxed:
    """Return a baseline transform with case-insensitive defaults."""
    return ExtractDiscreteAnswerFromBoxed(mapping={"Yes": 1, "No": 0}, missing_limit=0)


def test_call_single_string_returns_tensor(transform: ExtractDiscreteAnswerFromBoxed) -> None:
    """A single boxed string should yield an int tensor with trimmed, casefolded lookup."""
    result = transform("\\boxed{  YES  }")

    assert result.tolist() == [1]
    assert result.dtype == torch.long


def test_call_list_parses_code_fences(transform: ExtractDiscreteAnswerFromBoxed) -> None:
    """Lists of responses should parse markdown fenced boxed content and preserve order."""
    result = transform(["```latex\n\\boxed{No}\n```", "\\boxed{Yes}"])

    assert result.tolist() == [0, 1]


def test_call_ignores_surrounding_text(transform: ExtractDiscreteAnswerFromBoxed) -> None:
    """Noise around the boxed content should be ignored by extract_boxed."""
    raw_response = (
        "Let me think step by step:\n"
        "1. First consideration...\n"
        "2. Second point...\n"
        "\\boxed{Yes}\n"
        "That's my final answer!"
    )
    result = transform(raw_response)

    assert result.tolist() == [1]


def test_case_sensitive_behavior() -> None:
    """Case-sensitive mode should only match exact variants."""
    transform = ExtractDiscreteAnswerFromBoxed(
        mapping={"yes": 1}, case_sensitive=True, missing_limit=0
    )

    result = transform("\\boxed{yes}")
    assert result.tolist() == [1]
    with pytest.raises(ValueError, match=re.escape("Answer 'Yes' not found in mapping: ['yes']")):
        transform("\\boxed{Yes}")


def test_missing_answer_maps_to_fallback_when_allowed() -> None:
    """Missing answers should return the configured fallback when raising is disabled."""
    transform = ExtractDiscreteAnswerFromBoxed(
        mapping={"yes": 1},
        raise_if_missing=False,
        missing_answer=-42,
    )

    result = transform("\\boxed{maybe}")

    assert result.tolist() == [-42]


def test_missing_boxed_content_raises(transform: ExtractDiscreteAnswerFromBoxed) -> None:
    """Responses without boxed content should raise a descriptive error."""
    with pytest.raises(ValueError, match="Found 1 responses without valid structured data"):
        transform("My answer is Yes but not in boxed format")


def test_missing_limit_raises_after_threshold() -> None:
    """Missing boxed responses should respect the configured missing_limit."""
    transform = ExtractDiscreteAnswerFromBoxed(
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
        ExtractDiscreteAnswerFromBoxed(mapping={})


def test_multiple_boxed_expressions_returns_last() -> None:
    """When multiple boxed expressions exist, should use the last one."""
    transform = ExtractDiscreteAnswerFromBoxed(
        mapping={"A": 0, "B": 1}, missing_limit=0, raise_if_missing=False, missing_answer=-99
    )
    result = transform("First I think \\boxed{A} but wait, actually \\boxed{B}")

    # Should return the last boxed expression (B -> 1)
    assert result.tolist() == [1]


def test_boxed_with_letter_options(transform: ExtractDiscreteAnswerFromBoxed) -> None:
    """Boxed content with letter options (A, B, C, etc.) should work."""
    transform = ExtractDiscreteAnswerFromBoxed(mapping={"A": 0, "B": 1, "C": 2}, missing_limit=0)
    result = transform("\\boxed{B}")

    assert result.tolist() == [1]


def test_custom_answer_key() -> None:
    """Should use custom answer_key when specified."""
    # With custom answer_key, extract_boxed will return {"solution": "B"} instead of {"answer": "B"}
    # The base class will then look for structured_obj["solution"]
    transform = ExtractDiscreteAnswerFromBoxed(
        mapping={"A": 0, "B": 1, "C": 2}, answer_key="solution", missing_limit=0
    )
    result = transform("\\boxed{B}")

    # Should successfully extract using the custom key
    assert result.tolist() == [1]
