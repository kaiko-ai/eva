"""Tests for ExtractAnswerFromRaw post-processing transform."""

from eva.language.models.postprocess.extract_answer.raw.free_form import ExtractAnswerFromRaw


def test_single_string_returns_wrapped_in_list() -> None:
    """Single string input should be wrapped in a list unchanged."""
    transform = ExtractAnswerFromRaw()
    result = transform("Test response")

    assert result == ["Test response"]


def test_list_input_returns_unchanged() -> None:
    """List of strings should be returned unchanged."""
    transform = ExtractAnswerFromRaw()
    input_list = ["First", "Second", "Third"]
    result = transform(input_list)

    assert result == input_list


def test_accepts_optional_parameters() -> None:
    """Should accept optional parameters for compatibility."""
    transform = ExtractAnswerFromRaw(answer_key="custom", case_sensitive=False)
    result = transform("Test")

    assert result == ["Test"]
