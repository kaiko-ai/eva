"""Tests for XML text utilities."""

import pytest

from eva.language.utils.text.xml import extract_xml


@pytest.mark.parametrize(
    ("xml_str", "expected"),
    [
        # Basic extraction
        ("<answer>Yes</answer>", {"answer": "Yes"}),
        ("The correct answer is:\n<answer>Yes</answer> you agree?", {"answer": "Yes"}),
        (
            "<answer>No</answer><confidence>high</confidence>",
            {"answer": "No", "confidence": "high"},
        ),
        # Code fences
        ("```xml\n<answer>Yes</answer>\n```", {"answer": "Yes"}),
        ("```\n<answer>No</answer>\n```", {"answer": "No"}),
        (
            "Here's the answer:\n```xml\n<answer>Yes</answer>\n```\nThank you!",
            {"answer": "Yes"},
        ),
        # Whitespace handling
        ("<answer>  Yes  </answer>", {"answer": "Yes"}),
        # Empty text
        ("<answer></answer>", {"answer": ""}),
        # Special XML characters (& and <)
        (
            "<answer>H&E staining shows value <43</answer>",
            {"answer": "H&E staining shows value <43"},
        ),
    ],
    ids=[
        "simple",
        "surrounding_text_no_fence",
        "multiple_children",
        "markdown_code_fence",
        "plain_code_fence",
        "surrounding_text_with_fence",
        "whitespace",
        "empty_text",
        "special_xml_chars",
    ],
)
def test_extract_xml_valid_cases(xml_str: str, expected: dict) -> None:
    """Should extract valid XML in various formats."""
    result = extract_xml(xml_str)
    assert result == expected


@pytest.mark.parametrize(
    "xml_str",
    [
        "not valid xml",
    ],
    ids=["invalid_text"],
)
def test_extract_xml_invalid_returns_none(xml_str: str) -> None:
    """Invalid XML should return None when raise_if_missing is False."""
    result = extract_xml(xml_str, raise_if_missing=False)
    assert result is None


@pytest.mark.parametrize(
    "xml_str",
    [
        "not valid xml",
    ],
    ids=["invalid_text"],
)
def test_extract_xml_invalid_raises_when_configured(xml_str: str) -> None:
    """Invalid XML should raise ValueError when raise_if_missing is True."""
    with pytest.raises(ValueError, match="Failed to extract an XML object from the response"):
        extract_xml(xml_str, raise_if_missing=True)


def test_extract_xml_nested_elements() -> None:
    """Should handle nested elements by converting them to string."""
    xml_str = "<answer>Yes</answer><metadata><source>test</source></metadata>"
    result = extract_xml(xml_str)

    # Should get all tags at the top level
    assert isinstance(result, dict)
    assert "answer" in result
    assert result["answer"] == "Yes"
    assert "metadata" in result
