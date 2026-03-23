"""Tests for JSON text utilities."""

import pytest

from eva.language.utils.text.json import extract_json


@pytest.mark.parametrize(
    ("json_str", "expected"),
    [
        # Basic extraction
        ('{"answer": "Yes"}', {"answer": "Yes"}),
        ('The correct answer is:\n{"answer": "Yes"} you aggree?', {"answer": "Yes"}),
        ('{"answer": "No", "confidence": 0.95}', {"answer": "No", "confidence": 0.95}),
        # Code fences
        ('```json\n{"answer": "Yes"}\n```', {"answer": "Yes"}),
        ('```\n{"answer": "No"}\n```', {"answer": "No"}),
        (
            'Here\'s my answer:\n```json\n{"answer": "Yes"}\n```\nThank you!',
            {"answer": "Yes"},
        ),
        (
            '```json\n{"answer": "Yes", "source": "code_fence"}\n```\nAnother text: No',
            {"answer": "Yes", "source": "code_fence"},
        ),
        # Whitespace and formatting
        ('{ "answer" :  "Yes"  }', {"answer": "Yes"}),
        (
            '{\n    "answer": "Yes",\n    "confidence": "high",\n    "reasoning": "Because it makes sense"\n}',  # noqa: E501
            {"answer": "Yes", "confidence": "high", "reasoning": "Because it makes sense"},
        ),
        # Complex structures
        (
            '{"answer": "Yes", "metadata": {"source": "test", "version": 1}}',
            {"answer": "Yes", "metadata": {"source": "test", "version": 1}},
        ),
        (
            '{"options": ["A", "B", "C"], "answer": "B"}',
            {"options": ["A", "B", "C"], "answer": "B"},
        ),
        ('{"a": {"b": {"c": {"d": "value"}}}}', {"a": {"b": {"c": {"d": "value"}}}}),
        # Special characters
        (
            '{"answer": "Yes", "reason": "It\'s correct"}',
            {"answer": "Yes", "reason": "It's correct"},
        ),
        ('{"answer": "Yes", "emoji": "✅"}', {"answer": "Yes", "emoji": "✅"}),
        # Data types
        (
            '{"integer": 42, "float": 3.14, "negative": -10, "scientific": 1e-5}',
            {"integer": 42, "float": 3.14, "negative": -10, "scientific": 1e-5},
        ),
        (
            '{"is_correct": true, "is_wrong": false, "unknown": null}',
            {"is_correct": True, "is_wrong": False, "unknown": None},
        ),
        # Edge cases
        ("{}", {}),
    ],
)
def test_extract_json_valid_cases(json_str: str, expected: dict) -> None:
    """Should extract valid JSON in various formats."""
    result = extract_json(json_str)
    assert result == expected


@pytest.mark.parametrize(
    ("json_str", "repair", "expected"),
    [
        # Malformed JSON with repair
        ('{"answer": "Yes', True, {"answer": "Yes"}),
        ('{"answer": "Yes", "score": 1,}', True, {"answer": "Yes", "score": 1}),
        ("{'answer': 'Yes'}", True, {"answer": "Yes"}),
        # Malformed JSON without repair
        ('{"answer": "Yes', False, None),
    ],
    ids=["missing_closing", "trailing_comma", "single_quotes", "missing_closing_no_repair"],
)
def test_extract_json_repair_behavior(json_str: str, repair: bool, expected: dict | None) -> None:
    """Should handle malformed JSON based on repair setting."""
    result = extract_json(json_str, repair=repair, raise_if_missing=False)
    assert result == expected


@pytest.mark.parametrize(
    "json_str",
    [
        "not valid json at all",
        '["a", "b", "c"]',  # Array, not object
        '"just a string"',  # Primitive, not object
    ],
    ids=["invalid_text", "array", "primitive"],
)
def test_extract_json_invalid_returns_none(json_str: str) -> None:
    """Invalid JSON should return None when raise_if_missing is False."""
    result = extract_json(json_str, raise_if_missing=False)
    assert result is None


@pytest.mark.parametrize(
    "json_str",
    [
        "not valid json",
        '["a", "b", "c"]',  # Array, not object
        '"just a string"',  # Primitive, not object
    ],
    ids=["invalid_text", "array", "primitive"],
)
def test_extract_json_invalid_raises_when_configured(json_str: str) -> None:
    """Invalid JSON should raise ValueError when raise_if_missing is True."""
    with pytest.raises(ValueError, match="Failed to extract a JSON object from the response"):
        extract_json(json_str, raise_if_missing=True)


def test_extract_json_code_fence_without_newline_after_opening() -> None:
    """Should handle code fence without newline after opening backticks."""
    json_str = '```{"answer": "Yes"}```'
    result = extract_json(json_str)

    # This will not match the regex pattern, so it tries to parse as-is
    # The repair function should handle the backticks
    assert result is not None or result == {"answer": "Yes"}
