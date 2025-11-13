"""Tests for ExtractDiscreteAnswerFromRaw post-processing transform."""

import pytest
import torch

from eva.language.models.postprocess.extract_answer.raw.discrete import ExtractDiscreteAnswerFromRaw


@pytest.fixture
def transform() -> ExtractDiscreteAnswerFromRaw:
    """Return a baseline transform with case-insensitive defaults."""
    return ExtractDiscreteAnswerFromRaw(mapping={"Yes": 1, "No": 0}, missing_limit=0)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("After careful consideration, my answer is Yes", [1]),
        ("The correct answer is: No", [0]),
        ("My final answer is Yes.", [1]),
        ("I choose: No!", [0]),
        ('I select "No"', [0]),
    ],
)
def test_extract_answer_from_response(
    transform: ExtractDiscreteAnswerFromRaw, response: str, expected: list[int]
) -> None:
    """Should extract answers from various response formats."""
    result = transform(response)

    assert result.tolist() == expected
    assert result.dtype == torch.long


def test_call_list_preserves_order(transform: ExtractDiscreteAnswerFromRaw) -> None:
    """Lists of responses should preserve order."""
    result = transform(
        [
            "Reasoning here. Answer: No",
            "More reasoning. Final answer: Yes",
            "Analysis complete. No",
        ]
    )

    assert result.tolist() == [0, 1, 0]


def test_call_ignores_preceding_text(transform: ExtractDiscreteAnswerFromRaw) -> None:
    """Long explanations before the answer should be ignored."""
    raw_response = """
    Let me analyze this question step by step:
    1. First consideration...
    2. Second point...
    3. Final analysis...

    Based on all of the above, my answer is Yes.
    """
    result = transform(raw_response)

    assert result.tolist() == [1]


@pytest.mark.parametrize(
    "response",
    ["yes", "YES", "Yes", "yEs"],
)
def test_case_insensitive_matching(transform: ExtractDiscreteAnswerFromRaw, response: str) -> None:
    """Should match answers regardless of case by default."""
    result = transform(response)
    assert result.tolist() == [1]


def test_case_sensitive_behavior() -> None:
    """Case-sensitive mode should only match exact variants."""
    transform = ExtractDiscreteAnswerFromRaw(
        mapping={"yes": 1, "no": 0}, case_sensitive=True, missing_limit=0
    )

    result = transform("My answer is yes")
    assert result.tolist() == [1]
    with pytest.raises(ValueError, match="Found 1 responses without valid structured data"):
        transform("My answer is Yes")


def test_missing_answer_maps_to_fallback_when_allowed() -> None:
    """Missing answers should return the configured fallback when raising is disabled."""
    transform = ExtractDiscreteAnswerFromRaw(
        mapping={"yes": 1, "no": 0},
        raise_if_missing=False,
        missing_answer=-42,
    )

    result = transform("I don't know the answer to this question")

    assert result.tolist() == [-42]


def test_missing_answer_raises(transform: ExtractDiscreteAnswerFromRaw) -> None:
    """Responses without a valid answer should raise a descriptive error."""
    with pytest.raises(ValueError, match="Found 1 responses without valid structured data"):
        transform("This response has absolutely nothing valid in it")


def test_missing_limit_raises_after_threshold() -> None:
    """Missing responses should respect the configured missing_limit."""
    transform = ExtractDiscreteAnswerFromRaw(
        mapping={"no": 0, "yes": 1},
        missing_limit=3,
        missing_answer=-99,
    )
    result1 = transform("unknown")
    assert result1.tolist() == [-99]
    result2 = transform(["unknown", "unknown"])
    assert result2.tolist() == [-99, -99]
    with pytest.raises(ValueError, match="Found 4 responses without valid structured data"):
        transform("unknown")


def test_init_requires_non_empty_mapping() -> None:
    """An empty mapping should be rejected at construction time."""
    with pytest.raises(ValueError, match="`mapping` must be a non-empty dictionary."):
        ExtractDiscreteAnswerFromRaw(mapping={})


def test_supports_multi_word_keys() -> None:
    """Multi-word mapping keys should be supported and work correctly."""
    transform = ExtractDiscreteAnswerFromRaw(
        mapping={"Option A": 0, "Option B": 1, "True": 2, "False": 3}, missing_limit=0
    )

    test_cases = [
        ("The answer is Option A", [0]),
        ("I choose Option B", [1]),
        ("Answer: True", [2]),
        ("The correct choice is False", [3]),
    ]

    for text, expected in test_cases:
        result = transform(text)
        assert result.tolist() == expected, f"Failed for text: '{text}'"


def test_prioritizes_last_occurrence() -> None:
    """When multiple valid answers appear, should prioritize the last one."""
    transform = ExtractDiscreteAnswerFromRaw(
        mapping={"Yes": 1, "No": 0},
        missing_limit=0,
    )

    # "No" appears last within the lookback window
    result = transform("Initially I thought Yes but actually No")

    assert result.tolist() == [0]


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("Answer:   Yes  ", [1]),
        ("Answer:\nYes", [1]),
        ("Answer:\tNo", [0]),
    ],
)
def test_whitespace_handling(
    transform: ExtractDiscreteAnswerFromRaw, response: str, expected: list[int]
) -> None:
    """Should handle various whitespace patterns."""
    result = transform(response)
    assert result.tolist() == expected


@pytest.mark.parametrize(
    "response",
    ["", "   "],
)
def test_empty_string_returns_missing_answer(response: str) -> None:
    """Empty strings should be treated as missing answers."""
    transform = ExtractDiscreteAnswerFromRaw(
        mapping={"Yes": 1},
        raise_if_missing=False,
        missing_answer=-1,
    )

    result = transform(response)
    assert result.tolist() == [-1]


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("The correct answer is B", [1]),
        ("I choose option D.", [3]),
        ("My selection is A", [0]),
    ],
)
def test_answer_with_letter_options(response: str, expected: list[int]) -> None:
    """Should work with letter-based answers (A, B, C, D)."""
    transform = ExtractDiscreteAnswerFromRaw(
        mapping={"A": 0, "B": 1, "C": 2, "D": 3},
        missing_limit=0,
    )

    result = transform(response)
    assert result.tolist() == expected


def test_robust_to_similar_words() -> None:
    """Should only match exact answer options, not similar words."""
    # "yesterday" contains "yes" but shouldn't match
    transform = ExtractDiscreteAnswerFromRaw(
        mapping={"Yes": 1, "No": 0},
        raise_if_missing=False,
        missing_answer=-1,
    )

    result = transform("This happened yesterday")
    assert result.tolist() == [-1]
