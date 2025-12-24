"""Tests for ExtractAnswerFromJson post-processing transform."""

import pytest

from eva.language.models.postprocess.extract_answer.json.free_form import ExtractAnswerFromJson


@pytest.fixture
def transform() -> ExtractAnswerFromJson:
    """Return a baseline transform with default settings."""
    return ExtractAnswerFromJson()


def test_extract_basic_json_structure(transform: ExtractAnswerFromJson) -> None:
    """Basic JSON extraction should return the answer string."""
    result = transform('{"answer": "The capital of France is Paris."}')

    assert result == ["The capital of France is Paris."]


def test_extract_json_from_code_fence(transform: ExtractAnswerFromJson) -> None:
    """JSON wrapped in code fences should be extracted correctly."""
    json_response = '```json\n{"answer": "42", "confidence": "high"}\n```'
    result = transform(json_response)

    assert result == ["42"]


def test_custom_answer_key_extraction() -> None:
    """Custom answer keys should be respected in extraction."""
    transform = ExtractAnswerFromJson(answer_key="response")
    result = transform('{"response": "Custom key test", "other": "ignored"}')

    assert result == ["Custom key test"]


def test_malformed_json_returns_none(transform: ExtractAnswerFromJson) -> None:
    """Malformed JSON should return None when raise_if_missing is False."""
    result = transform("This is not JSON at all")

    assert result == [None]


def test_extract_list_preserves_order(transform: ExtractAnswerFromJson) -> None:
    """Lists of JSON responses should preserve order and extract all correctly."""
    json_list = [
        '{"answer": "First response"}',
        '{"answer": "Second response"}',
        '{"answer": "Third response"}',
    ]
    result = transform(json_list)

    assert result == [
        "First response",
        "Second response",
        "Third response",
    ]


def test_extract_json_ignores_surrounding_text(transform: ExtractAnswerFromJson) -> None:
    """JSON extraction should work with surrounding noise text."""
    noisy_response = 'Here\'s my reasoning...\n{"answer": "The correct answer"}\nThank you!'
    result = transform(noisy_response)

    assert result == ["The correct answer"]


def test_nested_json_objects(transform: ExtractAnswerFromJson) -> None:
    """Should handle nested JSON objects correctly."""
    nested_json = (
        '{"answer": "Main answer", "details": {"confidence": 0.95, "method": "reasoning"}}'
    )
    result = transform(nested_json)

    assert result == ["Main answer"]


def test_json_with_arrays(transform: ExtractAnswerFromJson) -> None:
    """Should handle JSON with array values."""
    json_with_array = (
        '{"answer": "Multiple choice", "options": ["A", "B", "C"], "selected": ["A", "C"]}'
    )
    result = transform(json_with_array)

    assert result == ["Multiple choice"]


def test_mixed_valid_invalid_json_responses() -> None:
    """Mixed batch with valid and invalid JSON should handle each appropriately."""
    transform = ExtractAnswerFromJson(raise_if_missing=False)
    responses = ['{"answer": "Valid JSON"}', "Not JSON at all", '{"answer": "Another valid"}']
    result = transform(responses)

    assert result == ["Valid JSON", None, "Another valid"]


def test_empty_json_values(transform: ExtractAnswerFromJson) -> None:
    """Empty values in JSON should be preserved."""
    result = transform('{"answer": "", "empty_field": null}')

    assert result == [""]


def test_plain_code_fence_without_language(transform: ExtractAnswerFromJson) -> None:
    """Should handle plain code fences without json language identifier."""
    result = transform('```\n{"answer": "Plain fence"}\n```')

    assert result == ["Plain fence"]


@pytest.mark.parametrize(
    ("return_dict", "input_value", "expected"),
    [
        (
            True,
            '{"answer": "The answer", "extra": "data"}',
            [{"answer": "The answer", "extra": "data"}],
        ),
        (False, '{"answer": "The answer", "extra": "data"}', ["The answer"]),
        (
            True,
            ['{"answer": "First"}', '{"answer": "Second"}'],
            [{"answer": "First"}, {"answer": "Second"}],
        ),
        (False, ['{"answer": "First"}', '{"answer": "Second"}'], ["First", "Second"]),
    ],
)
def test_return_dict(return_dict: bool, input_value: str | list, expected: list) -> None:
    """Should return dict or string based on return_dict parameter."""
    transform = ExtractAnswerFromJson(return_dict=return_dict)
    result = transform(input_value)

    assert result == expected


def test_return_dict_with_custom_answer_key() -> None:
    """Should use custom answer_key when specified with return_dict=True."""
    transform = ExtractAnswerFromJson(answer_key="response", return_dict=True)
    result = transform('{"response": "Custom key", "other": "ignored"}')

    assert result == [{"response": "Custom key", "other": "ignored"}]


def test_return_dict_with_missing() -> None:
    """Should return None for missing JSON content regardless of return_dict."""
    transform = ExtractAnswerFromJson(return_dict=True, raise_if_missing=False)
    result = transform("No JSON content here")

    assert result == [None]
