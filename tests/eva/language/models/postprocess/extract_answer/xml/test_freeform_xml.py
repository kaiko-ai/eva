"""Tests for ExtractAnswerFromXml post-processing transform."""

import pytest

from eva.language.models.postprocess.extract_answer.xml.free_form import ExtractAnswerFromXml


@pytest.fixture
def transform() -> ExtractAnswerFromXml:
    """Return a baseline transform with default settings."""
    return ExtractAnswerFromXml()


def test_extract_basic_xml_structure(transform: ExtractAnswerFromXml) -> None:
    """Basic XML extraction should return structured dictionary data."""
    result = transform("<answer>The capital of France is Paris.</answer>")

    assert result == [{"answer": "The capital of France is Paris."}]


def test_extract_xml_from_code_fence(transform: ExtractAnswerFromXml) -> None:
    """XML wrapped in code fences should be extracted correctly."""
    xml_response = "```xml\n<answer>42</answer>\n<confidence>high</confidence>\n```"
    result = transform(xml_response)

    assert result == [{"answer": "42", "confidence": "high"}]


def test_custom_answer_key_extraction() -> None:
    """Custom answer keys should be respected in extraction."""
    transform = ExtractAnswerFromXml(answer_key="response")
    result = transform("<response>Custom key test</response><other>ignored</other>")

    assert result == [{"response": "Custom key test", "other": "ignored"}]


def test_malformed_xml_returns_none(transform: ExtractAnswerFromXml) -> None:
    """Malformed XML should return None when raise_if_missing is False."""
    result = transform("This is not XML at all")

    assert result == [None]


def test_extract_list_preserves_order(transform: ExtractAnswerFromXml) -> None:
    """Lists of XML responses should preserve order and extract all correctly."""
    xml_list = [
        "<answer>First response</answer>",
        "<answer>Second response</answer>",
        "<answer>Third response</answer>",
    ]
    result = transform(xml_list)

    assert result == [
        {"answer": "First response"},
        {"answer": "Second response"},
        {"answer": "Third response"},
    ]


def test_extract_xml_ignores_surrounding_text(transform: ExtractAnswerFromXml) -> None:
    """XML extraction should work with surrounding noise text."""
    noisy_response = "Here's my reasoning...\n<answer>The correct answer</answer>\nThank you!"
    result = transform(noisy_response)

    assert result == [{"answer": "The correct answer"}]


def test_plain_code_fence_without_language(transform: ExtractAnswerFromXml) -> None:
    """Should handle plain code fences without xml language identifier."""
    result = transform("```\n<answer>Plain fence</answer>\n```")

    assert result == [{"answer": "Plain fence"}]


def test_multiple_xml_tags_extraction(transform: ExtractAnswerFromXml) -> None:
    """Should extract all XML tags present in the response."""
    xml_str = "<think>Step by step</think><answer>Answer</answer><confidence>0.95</confidence>"
    result = transform(xml_str)

    assert result == [{"think": "Step by step", "answer": "Answer", "confidence": "0.95"}]


def test_whitespace_in_values_is_stripped(transform: ExtractAnswerFromXml) -> None:
    """Leading and trailing whitespace in XML values should be stripped."""
    result = transform("<answer>\n  Whitespace answer  \n</answer>")

    assert result == [{"answer": "Whitespace answer"}]


def test_empty_xml_tags(transform: ExtractAnswerFromXml) -> None:
    """Empty XML tags should be handled gracefully."""
    result = transform("<answer></answer><empty></empty>")

    assert result == [{"answer": "", "empty": ""}]


def test_mixed_valid_invalid_responses() -> None:
    """Mixed batch with valid and invalid XML should handle each appropriately."""
    transform = ExtractAnswerFromXml(raise_if_missing=False)
    responses = ["<answer>Valid XML</answer>", "Not XML at all", "<answer>Another valid</answer>"]
    result = transform(responses)

    assert result == [{"answer": "Valid XML"}, None, {"answer": "Another valid"}]


def test_nested_xml_structure(transform: ExtractAnswerFromXml) -> None:
    """Should handle basic nested XML structures by flattening them."""
    nested_xml = "<answer><text>Nested content</text></answer>"
    result = transform(nested_xml)

    assert result == [{"answer": ""}]  # Inner content is not extracted in flat structure


def test_case_sensitivity_in_tags(transform: ExtractAnswerFromXml) -> None:
    """XML tag names should be case sensitive."""
    result = transform("<Answer>Case sensitive tag</Answer><answer>lowercase tag</answer>")

    assert result == [{"Answer": "Case sensitive tag", "answer": "lowercase tag"}]
