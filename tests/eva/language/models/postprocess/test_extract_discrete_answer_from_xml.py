"""Tests for ExtractDiscreteAnswerFromXml post-processing transform."""

import re

import pytest
import torch

from eva.language.models.postprocess.extract_answer.xml.discrete import ExtractDiscreteAnswerFromXml


@pytest.fixture
def transform() -> ExtractDiscreteAnswerFromXml:
    """Return a baseline transform with case-insensitive defaults."""
    return ExtractDiscreteAnswerFromXml(mapping={"Yes": 1, "No": 0}, missing_limit=0)


def test_call_single_string_returns_tensor(transform: ExtractDiscreteAnswerFromXml) -> None:
    """A single XML string should yield an int tensor with trimmed, casefolded lookup."""
    tensor = transform("<answer>  YES  </answer>")

    assert tensor.tolist() == [1]
    assert tensor.dtype == torch.long


def test_call_list_parses_code_fences(transform: ExtractDiscreteAnswerFromXml) -> None:
    """Lists of responses should parse markdown fenced XML and preserve order."""
    tensor = transform(["```xml\n<answer>No</answer>\n```", "<answer>Yes</answer>"])

    assert tensor.tolist() == [0, 1]


def test_call_ignores_surrounding_text(transform: ExtractDiscreteAnswerFromXml) -> None:
    """Noise around the XML blob should be ignored by extract_xml."""
    raw_response = "Final thoughts:\n" "```xml\n" "<answer>Yes</answer>\n" "```\n" "Thank you!"
    tensor = transform(raw_response)

    assert tensor.tolist() == [1]


def test_custom_answer_key_respected() -> None:
    """Custom answer_key should be used when extracting responses."""
    transform = ExtractDiscreteAnswerFromXml(mapping={"blue": 2}, answer_key="choice")
    tensor = transform("<choice>Blue</choice>")

    assert tensor.tolist() == [2]


def test_case_sensitive_behavior() -> None:
    """Case-sensitive mode should only match exact variants."""
    transform = ExtractDiscreteAnswerFromXml(
        mapping={"yes": 1}, case_sensitive=True, missing_limit=0
    )

    assert transform("<answer>yes</answer>").tolist() == [1]
    with pytest.raises(ValueError, match=re.escape("Answer 'Yes' not found in mapping: ['yes']")):
        transform("<answer>Yes</answer>")


def test_missing_answer_maps_to_fallback_when_allowed() -> None:
    """Missing answers should return the configured fallback when raising is disabled."""
    transform = ExtractDiscreteAnswerFromXml(
        mapping={"yes": 1},
        raise_if_missing=False,
        missing_response=-42,
    )

    tensor = transform("<answer>maybe</answer>")

    assert tensor.tolist() == [-42]


def test_missing_answer_key_raises(transform: ExtractDiscreteAnswerFromXml) -> None:
    """Responses without the answer key should raise a descriptive error."""
    with pytest.raises(ValueError, match="Found 1 responses without valid structured data"):
        transform("<not_answer>Yes</not_answer>")


def test_missing_limit_raises_after_threshold() -> None:
    """Missing XML responses should respect the configured missing_limit."""
    transform = ExtractDiscreteAnswerFromXml(
        mapping={"no": 0, "yes": 1},
        missing_limit=3,
        missing_response=-99,
    )
    assert transform("unknown").tolist() == [-99]
    assert transform(["unknown", "unknown"]).tolist() == [-99, -99]
    with pytest.raises(ValueError, match="Found 4 responses without valid structured data."):
        transform("unknown")


def test_init_requires_non_empty_mapping() -> None:
    """An empty mapping should be rejected at construction time."""
    with pytest.raises(ValueError, match="`mapping` must be a non-empty dictionary."):
        ExtractDiscreteAnswerFromXml(mapping={})


def test_multiple_tags_in_xml(transform: ExtractDiscreteAnswerFromXml) -> None:
    """XML with multiple tags should extract the answer tag correctly."""
    xml_str = "<reasoning>Because...</reasoning><answer>Yes</answer>"
    tensor = transform(xml_str)

    assert tensor.tolist() == [1]


def test_plain_code_fence_without_language(transform: ExtractDiscreteAnswerFromXml) -> None:
    """Should handle plain code fences without xml language identifier."""
    tensor = transform("```\n<answer>No</answer>\n```")

    assert tensor.tolist() == [0]


def test_whitespace_in_answer_is_stripped(transform: ExtractDiscreteAnswerFromXml) -> None:
    """Leading and trailing whitespace in answer values should be stripped."""
    tensor = transform("<answer>\n  Yes  \n</answer>")

    assert tensor.tolist() == [1]


def test_invalid_xml_returns_missing_response() -> None:
    """Invalid XML should return missing_response when raise_if_missing is False."""
    transform = ExtractDiscreteAnswerFromXml(
        mapping={"yes": 1, "no": 0},
        raise_if_missing=False,
        missing_response=-1,
    )

    tensor = transform("This is not XML at all")

    assert tensor.tolist() == [-1]
