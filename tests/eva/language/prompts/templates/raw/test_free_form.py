"""Tests for raw free-form prompt template."""

import pytest

from eva.language.prompts.templates import typings
from eva.language.prompts.templates.raw.free_form import RawFreeFormQuestionPromptTemplate


@pytest.fixture
def template() -> RawFreeFormQuestionPromptTemplate:
    """Return a baseline template instance for reuse across tests."""
    return RawFreeFormQuestionPromptTemplate()


def test_render_basic_trims_input(template: RawFreeFormQuestionPromptTemplate) -> None:
    """Renders the core prompt with minimal inputs and trims whitespace."""
    result = template.render(
        question="  What is the meaning of life?  ",
        context=None,
    )

    assert result.startswith("Question: What is the meaning of life?")
    assert "IMPORTANT: Respond in free form text, and make sure that your final answer" in result
    assert "Example Answer:" not in result


def test_render_context_formats_lists(template: RawFreeFormQuestionPromptTemplate) -> None:
    """Context lists should be bullet formatted."""
    result = template.render(
        question="Context handling?",
        context=[
            "First fact",
            "   Second fact   ",
            "Third fact",
        ],
    )

    # Extract the context block to confirm only non-empty entries remain.
    context_section = result.split("Context:\n", 1)[1].split("\n\n", 1)[0]
    context_lines = [line for line in context_section.splitlines() if line.startswith("- ")]
    assert context_lines == ["- First fact", "- Second fact", "- Third fact"]

    result_no_context = template.render(
        question="Skip context?",
        context=None,
    )
    assert "Context:" not in result_no_context


@pytest.mark.parametrize(
    ("enable_cot", "expected_fragment"),
    [
        (False, "Think step-by-step"),
        (True, "Think step-by-step"),
    ],
)
def test_render_enable_cot(enable_cot: bool, expected_fragment: str) -> None:
    """Prompt with enable_cot should contain a fragment asking the model to use thinking/CoT."""
    template = RawFreeFormQuestionPromptTemplate()
    result = template.render(
        question="Example answer?",
        context=None,
        enable_cot=enable_cot,
    )
    if enable_cot:
        assert expected_fragment in result
    else:
        assert expected_fragment not in result


def test_render_with_examples() -> None:
    """Template should render examples when provided."""
    template = RawFreeFormQuestionPromptTemplate()
    examples = [
        typings.QuestionAnswerExample(
            question="What is 2+2?",
            answer="4",
        ),
        typings.QuestionAnswerExample(
            question="What color is the sky?",
            answer="Blue",
        ),
    ]

    result = template.render(
        question="Test question",
        context=None,
        examples=examples,
    )

    assert "Below are some examples of how to answer questions:" in result
    assert "Example 1:" in result
    assert "What is 2+2?" in result
    assert "Answer: 4" in result
    assert "Example 2:" in result
    assert "What color is the sky?" in result
    assert "Answer: Blue" in result
    assert "Now please answer the following question." in result

    # Should not show default example format when examples are provided
    assert "Example Answer:" not in result


def test_render_without_examples() -> None:
    """Template should not show example format when no examples or example_answer provided."""
    template = RawFreeFormQuestionPromptTemplate()

    result = template.render(
        question="Test question",
        context=None,
    )
    assert "Example Answer:" not in result
    assert "Below are some examples:" not in result


def test_render_with_preamble() -> None:
    """Template should include preamble when provided."""
    template = RawFreeFormQuestionPromptTemplate()
    preamble = "You are a helpful assistant."

    result = template.render(
        question="Test question",
        context=None,
        preamble=preamble,
    )

    assert result.startswith("You are a helpful assistant.")


@pytest.mark.parametrize("question", ["", "   ", 123])
def test_render_invalid_question_raises_error(
    template: RawFreeFormQuestionPromptTemplate, question: object
) -> None:
    """Invalid questions should raise a descriptive ValueError."""
    with pytest.raises(ValueError, match="`question` must be a non-empty string"):
        template.render(
            question=question,  # type: ignore[arg-type]
            context=None,
        )


def test_render_instance_enable_cot() -> None:
    """Template instance with enable_cot should always include CoT instruction."""
    template = RawFreeFormQuestionPromptTemplate()
    result = template.render(
        question="Test question",
        context=None,
        enable_cot=True,
    )

    assert "Think step-by-step" in result


def test_render_with_example_answer() -> None:
    """Template should show example answer when provided."""
    template = RawFreeFormQuestionPromptTemplate()

    result = template.render(
        question="Test question",
        context=None,
        example_answer="42",
    )

    assert "Example Answer:" in result
    assert "42" in result
    assert "First, provide your reasoning" not in result


def test_render_with_example_answer_and_cot() -> None:
    """Template shows CoT-formatted example when example_answer and enable_cot are provided."""
    template = RawFreeFormQuestionPromptTemplate()

    result = template.render(
        question="Test question",
        context=None,
        example_answer="42",
        enable_cot=True,
    )

    assert "Example Answer:" in result
    assert "First, provide your reasoning for why you chose this answer here..." in result
    assert "Then, provide your final answer: 42" in result
    assert "Think step-by-step" in result
