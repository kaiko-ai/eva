"""Tests for JSON multiple choice prompt template."""

import pytest

from eva.language.prompts.templates.json.multiple_choice import JsonMultipleChoicePromptTemplate


def test_render_basic():
    """Test basic rendering with minimal parameters."""
    template = JsonMultipleChoicePromptTemplate()
    result = template.render(
        question="What is the capital of France?",
        context=None,
        answer_options=["Paris", "London", "Berlin"],
    )

    assert "What is the capital of France?" in result
    assert "- Paris" in result
    assert "- London" in result
    assert "- Berlin" in result
    assert '"answer":' in result
    assert '"reason":' in result


def test_render_with_context_string():
    """Test rendering with a single context string."""
    template = JsonMultipleChoicePromptTemplate()
    result = template.render(
        question="What is the capital?",
        context="France is a country in Europe.",
        answer_options=["Paris", "London"],
    )

    assert "Context:" in result
    assert "- France is a country in Europe." in result


def test_render_with_context_list():
    """Test rendering with multiple context strings."""
    template = JsonMultipleChoicePromptTemplate()
    result = template.render(
        question="What is the capital?",
        context=["France is a country.", "Paris is a city."],
        answer_options=["Paris", "London"],
    )

    assert "Context:" in result
    assert "- France is a country." in result
    assert "- Paris is a city." in result


def test_render_with_option_letters():
    """Test rendering with lettered options."""
    template = JsonMultipleChoicePromptTemplate(use_option_letters=True)
    result = template.render(
        question="Pick a color",
        context=None,
        answer_options=["Red", "Blue", "Green"],
    )

    assert "A. Red" in result
    assert "B. Blue" in result
    assert "C. Green" in result
    assert 'The value for "answer" must be the letter' in result


def test_render_without_option_letters():
    """Test rendering with bullet point options."""
    template = JsonMultipleChoicePromptTemplate(use_option_letters=False)
    result = template.render(
        question="Pick a color",
        context=None,
        answer_options=["Red", "Blue"],
    )

    assert "- Red" in result
    assert "- Blue" in result
    assert 'The value for "answer" must exactly match one of the options' in result


def test_render_with_custom_keys():
    """Test rendering with custom answer and reason keys."""
    template = JsonMultipleChoicePromptTemplate(
        answer_key="choice",
        reason_key="explanation",
    )
    result = template.render(
        question="Test question",
        context=None,
        answer_options=["A", "B"],
    )

    assert '"choice":' in result
    assert '"explanation":' in result
    assert '"answer":' not in result
    assert '"reason":' not in result


def test_render_with_example_answer():
    """Test rendering with custom example answer."""
    template = JsonMultipleChoicePromptTemplate()
    result = template.render(
        question="Test question",
        context=None,
        answer_options=["Option A", "Option B"],
        example_answer="Option B",
    )

    assert '"answer": "Option B"' in result


def test_render_with_example_reason():
    """Test rendering with custom example reason."""
    template = JsonMultipleChoicePromptTemplate()
    result = template.render(
        question="Test question",
        context=None,
        answer_options=["A", "B"],
        example_reason="This is the correct choice because...",
    )

    assert '"reason": "This is the correct choice because..."' in result


def test_render_with_preamble():
    """Test rendering with preamble text."""
    template = JsonMultipleChoicePromptTemplate()
    result = template.render(
        question="Test question",
        context=None,
        answer_options=["A", "B"],
        preamble="You are a helpful assistant.",
    )

    assert result.startswith("You are a helpful assistant.")


def test_render_strips_whitespace():
    """Test that rendering strips excessive whitespace."""
    template = JsonMultipleChoicePromptTemplate()
    result = template.render(
        question="  Question with spaces  ",
        context="  Context with spaces  ",
        answer_options=["  Option A  ", "  Option B  "],
    )

    assert "Question with spaces" in result
    assert "Context with spaces" in result
    assert "  Question with spaces  " not in result


def test_render_empty_question_raises_error():
    """Test that empty question raises ValueError."""
    template = JsonMultipleChoicePromptTemplate()

    with pytest.raises(ValueError, match="`question` must be a non-empty string"):
        template.render(
            question="",
            context=None,
            answer_options=["A", "B"],
        )


def test_render_whitespace_only_question_raises_error():
    """Test that whitespace-only question raises ValueError."""
    template = JsonMultipleChoicePromptTemplate()

    with pytest.raises(ValueError, match="`question` must be a non-empty string"):
        template.render(
            question="   ",
            context=None,
            answer_options=["A", "B"],
        )


def test_render_non_string_question_raises_error():
    """Test that non-string question raises ValueError."""
    template = JsonMultipleChoicePromptTemplate()

    with pytest.raises(ValueError, match="`question` must be a non-empty string"):
        template.render(
            question=123,  # type: ignore
            context=None,
            answer_options=["A", "B"],
        )


def test_default_example_answer_without_letters():
    """Test that default example answer is first option without letters."""
    template = JsonMultipleChoicePromptTemplate(use_option_letters=False)
    result = template.render(
        question="Test",
        context=None,
        answer_options=["First", "Second"],
    )

    assert '"answer": "First"' in result


def test_default_example_answer_with_letters():
    """Test that default example answer is 'A' with letters."""
    template = JsonMultipleChoicePromptTemplate(use_option_letters=True)
    result = template.render(
        question="Test",
        context=None,
        answer_options=["First", "Second"],
    )

    assert '"answer": "A"' in result
