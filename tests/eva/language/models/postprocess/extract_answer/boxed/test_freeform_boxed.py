"""Tests for ExtractAnswerFromBoxed post-processing transform."""

import pytest

from eva.language.models.postprocess.extract_answer.boxed.free_form import ExtractAnswerFromBoxed


@pytest.fixture
def transform() -> ExtractAnswerFromBoxed:
    """Return a baseline transform with default settings."""
    return ExtractAnswerFromBoxed()


def test_extract_basic_boxed_structure(transform: ExtractAnswerFromBoxed) -> None:
    """Basic boxed extraction should return the answer string."""
    result = transform("\\boxed{The capital of France is Paris.}")

    assert result == ["The capital of France is Paris."]


def test_extract_boxed_from_code_fence(transform: ExtractAnswerFromBoxed) -> None:
    """Boxed content wrapped in code fences should be extracted correctly."""
    boxed_response = "```latex\n\\boxed{42}\n```"
    result = transform(boxed_response)

    assert result == ["42"]


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
        "First response",
        "Second response",
        "Third response",
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

    assert result == ["The correct answer"]


def test_boxed_with_math_expression(transform: ExtractAnswerFromBoxed) -> None:
    """Should handle boxed mathematical expressions."""
    math_response = "The derivative is \\boxed{2x + 3}"
    result = transform(math_response)

    assert result == ["2x + 3"]


def test_boxed_with_single_letter(transform: ExtractAnswerFromBoxed) -> None:
    """Should handle boxed single letter content."""
    result = transform("After careful consideration, \\boxed{B}")

    assert result == ["B"]


def test_mixed_valid_invalid_boxed_responses() -> None:
    """Mixed batch with valid and invalid boxed content should handle each appropriately."""
    transform = ExtractAnswerFromBoxed(raise_if_missing=False)
    responses = ["\\boxed{Valid boxed}", "Not boxed at all", "\\boxed{Another valid}"]
    result = transform(responses)

    assert result == ["Valid boxed", None, "Another valid"]


def test_boxed_with_whitespace(transform: ExtractAnswerFromBoxed) -> None:
    """Boxed content with leading/trailing whitespace should be trimmed."""
    result = transform("\\boxed{  answer with spaces  }")

    assert result == ["answer with spaces"]


def test_multiple_boxed_expressions_returns_last(transform: ExtractAnswerFromBoxed) -> None:
    """When multiple boxed expressions exist, should return the last one."""
    result = transform("Initially I thought \\boxed{wrong answer} but actually \\boxed{correct}")

    # Should return the last boxed expression
    assert result == ["correct"]


def test_boxed_with_latex_code_fence(transform: ExtractAnswerFromBoxed) -> None:
    """Should handle latex code fence."""
    result = transform("```latex\n\\boxed{x^2 + y^2 = r^2}\n```")

    assert result == ["x^2 + y^2 = r^2"]


def test_boxed_with_math_code_fence(transform: ExtractAnswerFromBoxed) -> None:
    """Should handle math code fence."""
    result = transform("```math\n\\boxed{\\pi r^2}\n```")

    assert result == ["\\pi r^2"]


def test_plain_code_fence_with_boxed(transform: ExtractAnswerFromBoxed) -> None:
    """Should handle plain code fences without language identifier."""
    result = transform("```\n\\boxed{Plain fence answer}\n```")

    assert result == ["Plain fence answer"]


def test_boxed_multiline_content(transform: ExtractAnswerFromBoxed) -> None:
    """Should handle boxed content spanning multiple lines (though unusual)."""
    # Note: The regex uses non-greedy match with DOTALL flag
    result = transform("\\boxed{Line 1\nLine 2}")

    assert result == ["Line 1\nLine 2"]


def test_boxed_with_nested_braces_simple(transform: ExtractAnswerFromBoxed) -> None:
    """Should handle boxed content with nested braces like LaTeX commands."""
    result = transform("The answer is \\boxed{\\frac{1}{2}}")

    assert result == ["\\frac{1}{2}"]


def test_boxed_with_nested_braces_complex(transform: ExtractAnswerFromBoxed) -> None:
    """Should handle boxed content with multiple levels of nested braces."""
    result = transform("\\boxed{\\sqrt{\\frac{a^{2} + b^{2}}{c}}}")

    assert result == ["\\sqrt{\\frac{a^{2} + b^{2}}{c}}"]


def test_boxed_with_nested_braces_and_text(transform: ExtractAnswerFromBoxed) -> None:
    """Should handle nested braces with mixed text and LaTeX."""
    result = transform("The solution is \\boxed{x = \\frac{-b \\pm \\sqrt{b^{2} - 4ac}}{2a}}")

    assert result == ["x = \\frac{-b \\pm \\sqrt{b^{2} - 4ac}}{2a}"]


def test_multiple_boxed_with_nested_braces(transform: ExtractAnswerFromBoxed) -> None:
    """Should return last boxed when multiple exist with nested braces."""
    result = transform("First try: \\boxed{\\frac{1}{3}} but correct is \\boxed{\\frac{2}{3}}")

    assert result == ["\\frac{2}{3}"]


def test_boxed_with_multiple_entries_takes_last(transform: ExtractAnswerFromBoxed) -> None:
    """Should use the last boxed entry when model refines its answer."""
    response = (
        "Let me think step by step.\n"
        "Initially, I calculated \\boxed{42}\n"
        "Wait, let me recalculate...\n"
        "Actually, the correct answer is \\boxed{84}"
    )
    result = transform(response)

    assert result == ["84"]


def test_boxed_nested_set_notation(transform: ExtractAnswerFromBoxed) -> None:
    """Should handle set notation with nested braces."""
    result = transform("\\boxed{\\{x \\in \\mathbb{R} : x > 0\\}}")

    assert result == ["\\{x \\in \\mathbb{R} : x > 0\\}"]


def test_custom_answer_key() -> None:
    """Should use custom answer_key when specified with return_dict=True."""
    transform = ExtractAnswerFromBoxed(answer_key="solution", return_dict=True)
    result = transform("\\boxed{42}")

    assert result == [{"solution": "42"}]


@pytest.mark.parametrize(
    ("return_dict", "input_value", "expected"),
    [
        (True, "\\boxed{The answer}", [{"answer": "The answer"}]),
        (False, "\\boxed{The answer}", ["The answer"]),
        (True, ["\\boxed{First}", "\\boxed{Second}"], [{"answer": "First"}, {"answer": "Second"}]),
        (False, ["\\boxed{First}", "\\boxed{Second}"], ["First", "Second"]),
    ],
)
def test_return_dict(return_dict: bool, input_value: str | list, expected: list) -> None:
    """Should return dict or string based on return_dict parameter."""
    transform = ExtractAnswerFromBoxed(return_dict=return_dict)
    result = transform(input_value)

    assert result == expected


def test_return_dict_with_missing() -> None:
    """Should return None for missing boxed content regardless of return_dict."""
    transform = ExtractAnswerFromBoxed(return_dict=True, raise_if_missing=False)
    result = transform("No boxed content here")

    assert result == [None]
