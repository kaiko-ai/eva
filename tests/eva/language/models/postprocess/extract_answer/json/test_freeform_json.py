"""Tests for ExtractAnswerFromJson post-processing transform."""

import pytest

from eva.language.models.postprocess.extract_answer.json.free_form import ExtractAnswerFromJson


@pytest.fixture
def transform() -> ExtractAnswerFromJson:
    """Return a baseline transform with default settings."""
    return ExtractAnswerFromJson()


def test_extract_basic_json_structure(transform: ExtractAnswerFromJson) -> None:
    """Basic JSON extraction should return structured dictionary data."""
    result = transform('{"answer": "The capital of France is Paris."}')

    assert result == [{"answer": "The capital of France is Paris."}]


def test_extract_json_from_code_fence(transform: ExtractAnswerFromJson) -> None:
    """JSON wrapped in code fences should be extracted correctly."""
    json_response = '```json\n{"answer": "42", "confidence": "high"}\n```'
    result = transform(json_response)

    assert result == [{"answer": "42", "confidence": "high"}]


def test_custom_answer_key_extraction() -> None:
    """Custom answer keys should be respected in extraction."""
    transform = ExtractAnswerFromJson(answer_key="response")
    result = transform('{"response": "Custom key test", "other": "ignored"}')

    assert result == [{"response": "Custom key test", "other": "ignored"}]


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
        {"answer": "First response"},
        {"answer": "Second response"},
        {"answer": "Third response"},
    ]


def test_extract_json_ignores_surrounding_text(transform: ExtractAnswerFromJson) -> None:
    """JSON extraction should work with surrounding noise text."""
    noisy_response = 'Here\'s my reasoning...\n{"answer": "The correct answer"}\nThank you!'
    result = transform(noisy_response)

    assert result == [{"answer": "The correct answer"}]


def test_nested_json_objects(transform: ExtractAnswerFromJson) -> None:
    """Should handle nested JSON objects correctly."""
    nested_json = (
        '{"answer": "Main answer", "details": {"confidence": 0.95, "method": "reasoning"}}'
    )
    result = transform(nested_json)

    expected = {"answer": "Main answer", "details": {"confidence": 0.95, "method": "reasoning"}}
    assert result == [expected]


def test_json_with_arrays(transform: ExtractAnswerFromJson) -> None:
    """Should handle JSON with array values."""
    json_with_array = (
        '{"answer": "Multiple choice", "options": ["A", "B", "C"], "selected": ["A", "C"]}'
    )
    result = transform(json_with_array)

    expected = {"answer": "Multiple choice", "options": ["A", "B", "C"], "selected": ["A", "C"]}
    assert result == [expected]


def test_mixed_valid_invalid_json_responses() -> None:
    """Mixed batch with valid and invalid JSON should handle each appropriately."""
    transform = ExtractAnswerFromJson(raise_if_missing=False)
    responses = ['{"answer": "Valid JSON"}', "Not JSON at all", '{"answer": "Another valid"}']
    result = transform(responses)

    assert result == [{"answer": "Valid JSON"}, None, {"answer": "Another valid"}]


def test_empty_json_values(transform: ExtractAnswerFromJson) -> None:
    """Empty values in JSON should be preserved."""
    result = transform('{"answer": "", "empty_field": null}')

    assert result == [{"answer": "", "empty_field": None}]


def test_plain_code_fence_without_language(transform: ExtractAnswerFromJson) -> None:
    """Should handle plain code fences without json language identifier."""
    result = transform('```\n{"answer": "Plain fence"}\n```')

    assert result == [{"answer": "Plain fence"}]
