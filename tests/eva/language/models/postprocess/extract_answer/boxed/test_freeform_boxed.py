"""Tests for ExtractAnswerFromBoxed post-processing transform."""

import pytest

from eva.language.models.postprocess.extract_answer.boxed.free_form import ExtractAnswerFromBoxed


@pytest.fixture
def transform() -> ExtractAnswerFromBoxed:
    """Return a baseline transform with default settings."""
    return ExtractAnswerFromBoxed()


def test_extract_basic_boxed_structure(transform: ExtractAnswerFromBoxed) -> None:
    """Basic boxed extraction should return structured dictionary data."""
    result = transform("\\boxed{The capital of France is Paris.}")

    assert result == [{"answer": "The capital of France is Paris."}]


def test_extract_boxed_from_code_fence(transform: ExtractAnswerFromBoxed) -> None:
    """Boxed content wrapped in code fences should be extracted correctly."""
    boxed_response = "```latex\n\\boxed{42}\n```"
    result = transform(boxed_response)

    assert result == [{"answer": "42"}]


def test_malformed_boxed_returns_none(transform: ExtractAnswerFromBoxed) -> None:
    """Response without boxed content should return None when raise_if_missing is False."""
    result = transform("This is not boxed at all")

    assert result == [None]


def test_extract_list_preserves_order(transform: ExtractAnswerFromBoxed) -> None:
    """Lists of boxed responses should preserve order and extract all correctly."""
    boxed_list = [
        "\\boxed{First response}",
        "\\boxed{Second response}",
        "\\boxed{Third response}",
    ]
    result = transform(boxed_list)

    assert result == [
        {"answer": "First response"},
        {"answer": "Second response"},
        {"answer": "Third response"},
    ]


def test_extract_boxed_ignores_surrounding_text(transform: ExtractAnswerFromBoxed) -> None:
    """Boxed extraction should work with surrounding reasoning text."""
    noisy_response = (
        "Here's my reasoning...\n"
        "Step 1: Consider the problem\n"
        "Step 2: Apply the formula\n"
        "\\boxed{The correct answer}\n"
        "Thank you!"
    )
    result = transform(noisy_response)

    assert result == [{"answer": "The correct answer"}]


def test_boxed_with_math_expression(transform: ExtractAnswerFromBoxed) -> None:
    """Should handle boxed mathematical expressions."""
    math_response = "The derivative is \\boxed{2x + 3}"
    result = transform(math_response)

    assert result == [{"answer": "2x + 3"}]


def test_boxed_with_single_letter(transform: ExtractAnswerFromBoxed) -> None:
    """Should handle boxed single letter content."""
    result = transform("After careful consideration, \\boxed{B}")

    assert result == [{"answer": "B"}]


def test_mixed_valid_invalid_boxed_responses() -> None:
    """Mixed batch with valid and invalid boxed content should handle each appropriately."""
    transform = ExtractAnswerFromBoxed(raise_if_missing=False)
    responses = ["\\boxed{Valid boxed}", "Not boxed at all", "\\boxed{Another valid}"]
    result = transform(responses)

    assert result == [{"answer": "Valid boxed"}, None, {"answer": "Another valid"}]


def test_boxed_with_whitespace(transform: ExtractAnswerFromBoxed) -> None:
    """Boxed content with leading/trailing whitespace should be trimmed."""
    result = transform("\\boxed{  answer with spaces  }")

    assert result == [{"answer": "answer with spaces"}]


def test_multiple_boxed_expressions_returns_none(transform: ExtractAnswerFromBoxed) -> None:
    """When multiple boxed expressions exist, should return None (invalid response)."""
    result = transform("Initially I thought \\boxed{wrong answer} but actually \\boxed{correct}")

    # Should return None since we can't determine which answer is correct
    assert result == [None]


def test_boxed_with_latex_code_fence(transform: ExtractAnswerFromBoxed) -> None:
    """Should handle latex code fence."""
    result = transform("```latex\n\\boxed{x^2 + y^2 = r^2}\n```")

    assert result == [{"answer": "x^2 + y^2 = r^2"}]


def test_boxed_with_math_code_fence(transform: ExtractAnswerFromBoxed) -> None:
    """Should handle math code fence."""
    result = transform("```math\n\\boxed{\\pi r^2}\n```")

    assert result == [{"answer": "\\pi r^2"}]


def test_plain_code_fence_with_boxed(transform: ExtractAnswerFromBoxed) -> None:
    """Should handle plain code fences without language identifier."""
    result = transform("```\n\\boxed{Plain fence answer}\n```")

    assert result == [{"answer": "Plain fence answer"}]


def test_boxed_multiline_content(transform: ExtractAnswerFromBoxed) -> None:
    """Should handle boxed content spanning multiple lines (though unusual)."""
    # Note: The regex uses non-greedy match with DOTALL flag
    result = transform("\\boxed{Line 1\nLine 2}")

    assert result == [{"answer": "Line 1\nLine 2"}]
