"""Tests for XML free-form prompt template."""

import pytest

from eva.language.prompts.templates import typings
from eva.language.prompts.templates.xml.free_form import XmlFreeFormPromptTemplate


@pytest.fixture
def template() -> XmlFreeFormPromptTemplate:
    """Return a baseline template instance for reuse across tests."""
    return XmlFreeFormPromptTemplate()


def test_render_basic_trims_input(template: XmlFreeFormPromptTemplate) -> None:
    """Renders the core prompt with minimal inputs and trims whitespace."""
    result = template.render(
        question="  What is the meaning of life?  ",
        context=None,
    )

    assert result.startswith("Question: What is the meaning of life?")
    assert "IMPORTANT: Provide your final answer within <answer></answer> tags" in result
    assert "<answer>" in result
    assert "</answer>" in result
    assert "Example Answer:" in result


def test_render_context_formats_lists(template: XmlFreeFormPromptTemplate) -> None:
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


@pytest.mark.parametrize(("answer_key"), ["choice", "response"])
def test_render_custom_keys_respected(answer_key: str) -> None:
    """Custom answer_key propagate to the instructions and example XML."""
    template = XmlFreeFormPromptTemplate()
    result = template.render(
        question="Test question",
        context=None,
        answer_key=answer_key,
    )

    assert f"<{answer_key}>" in result
    assert f"</{answer_key}>" in result
    assert "<answer>" not in result or answer_key == "answer"


@pytest.mark.parametrize(
    ("enable_cot", "expected_fragment"),
    [
        (False, "Think step-by-step"),
        (True, "Think step-by-step"),
    ],
)
def test_render_enable_cot(enable_cot: bool, expected_fragment: str) -> None:
    """Prompt with enable_cot should contain a fragment asking the model to use thinking/CoT."""
    template = XmlFreeFormPromptTemplate()
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
    template = XmlFreeFormPromptTemplate()
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

    assert "Below are some examples:" in result
    assert "Example 1:" in result
    assert "What is 2+2?" in result
    assert "Answer: 4" in result
    assert "Example 2:" in result
    assert "What color is the sky?" in result
    assert "Answer: Blue" in result
    assert "Now please answer the initial question." in result

    # Should not show default example format when examples are provided
    assert "Example Answer:" not in result


def test_render_without_examples() -> None:
    """Template should show default example format when no examples provided."""
    template = XmlFreeFormPromptTemplate()

    result = template.render(
        question="Test question",
        context=None,
    )

    assert "Example Answer:" in result
    assert "Your explanation for why you chose this answer can go here..." in result
    assert "Below are some examples:" not in result


def test_render_with_preamble() -> None:
    """Template should include preamble when provided."""
    template = XmlFreeFormPromptTemplate()
    preamble = "You are a helpful assistant."

    result = template.render(
        question="Test question",
        context=None,
        preamble=preamble,
    )

    assert result.startswith("You are a helpful assistant.")


@pytest.mark.parametrize("question", ["", "   ", 123])
def test_render_invalid_question_raises_error(
    template: XmlFreeFormPromptTemplate, question: object
) -> None:
    """Invalid questions should raise a descriptive ValueError."""
    with pytest.raises(ValueError, match="`question` must be a non-empty string"):
        template.render(
            question=question,  # type: ignore[arg-type]
            context=None,
        )


def test_render_enable_cot_with_examples_hides_instruction() -> None:
    """When examples are provided, CoT instruction should be hidden even if enable_cot is True."""
    template = XmlFreeFormPromptTemplate()
    examples = [
        typings.QuestionAnswerExample(
            question="Test?",
            answer="Yes",
        ),
    ]

    result = template.render(
        question="Test question",
        context=None,
        examples=examples,
        enable_cot=True,
    )

    # CoT instruction should not appear when examples are provided
    assert "Think step-by-step" not in result
    assert "Below are some examples:" in result
