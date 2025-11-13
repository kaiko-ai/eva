"""Tests for XML multiple choice prompt template."""

import pytest

from eva.language.prompts.templates.xml.multiple_choice import XmlMultipleChoicePromptTemplate


@pytest.fixture
def template() -> XmlMultipleChoicePromptTemplate:
    """Return a baseline template instance for reuse across tests."""
    return XmlMultipleChoicePromptTemplate()


def test_render_basic_trims_input(template: XmlMultipleChoicePromptTemplate) -> None:
    """Renders the core prompt with minimal inputs and trims whitespace."""
    result = template.render(
        question="  What is the capital of France?  ",
        context=None,
        answer_options=["  Paris  ", "London"],
    )

    assert result.startswith("Question: What is the capital of France?")
    assert "- Paris" in result
    assert "- London" in result
    assert "<answer>" in result
    assert "</answer>" in result


def test_render_context_formats_lists(template: XmlMultipleChoicePromptTemplate) -> None:
    """Context lists should be bullet formatted."""
    result = template.render(
        question="Context handling?",
        context=[
            "First fact",
            "   Second fact   ",
            "Third fact",
        ],
        answer_options=["Yes", "No"],
    )

    # Extract the context block to confirm only non-empty entries remain.
    context_section = result.split("Context:\n", 1)[1].split("\n\n", 1)[0]
    context_lines = [line for line in context_section.splitlines() if line.startswith("- ")]
    assert context_lines == ["- First fact", "- Second fact", "- Third fact"]

    result_no_context = template.render(
        question="Skip context?",
        context=None,
        answer_options=["Yes", "No"],
    )
    assert "Context:" not in result_no_context


@pytest.mark.parametrize(
    ("use_letters", "expected_option", "instruction_snippet"),
    [
        (True, "A. Red", "must be the letter"),
        (False, "- Red", "must exactly match"),
    ],
)
def test_render_option_styles(
    use_letters: bool, expected_option: str, instruction_snippet: str
) -> None:
    """Rendering switches between lettered options and bullet options."""
    template = XmlMultipleChoicePromptTemplate()
    result = template.render(
        question="Pick a color",
        context=None,
        answer_options=["Red", "Blue"],
        use_option_letters=use_letters,
    )

    assert expected_option in result
    assert instruction_snippet in result


@pytest.mark.parametrize(("answer_key"), ["choice", "response"])
def test_render_custom_keys_respected(answer_key: str) -> None:
    """Custom answer_key propagate to the instructions and example XML."""
    template = XmlMultipleChoicePromptTemplate()
    result = template.render(
        question="Test question",
        context=None,
        answer_options=["A", "B"],
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
    template = XmlMultipleChoicePromptTemplate()
    result = template.render(
        question="Example answer?",
        context=None,
        answer_options=["First", "Second"],
        enable_cot=enable_cot,
    )
    if enable_cot:
        assert expected_fragment in result
    else:
        assert expected_fragment not in result


@pytest.mark.parametrize("question", ["", "   ", 123])
def test_render_invalid_question_raises_error(
    template: XmlMultipleChoicePromptTemplate, question: object
) -> None:
    """Invalid questions should raise a descriptive ValueError."""
    with pytest.raises(ValueError, match="`question` must be a non-empty string"):
        template.render(
            question=question,  # type: ignore[arg-type]
            context=None,
            answer_options=["A", "B"],
        )


@pytest.mark.parametrize(
    "answer_options",
    [
        [],
        ["   "],
        ["Valid", ""],
    ],
)
def test_render_invalid_answer_options_raises_error(
    template: XmlMultipleChoicePromptTemplate, answer_options: list[object]
) -> None:
    """Invalid answer options should raise a descriptive ValueError."""
    with pytest.raises(ValueError, match="`items` must be all non-empty strings"):
        template.render(
            question="Options?",
            context=None,
            answer_options=answer_options,  # type: ignore[arg-type]
        )


def test_render_with_too_many_lettered_options() -> None:
    """Using option letters should enforce the alphabet upper bound."""
    template = XmlMultipleChoicePromptTemplate()
    answer_options = [f"Option {i}" for i in range(27)]

    with pytest.raises(ValueError, match="Maximum 26 items supported for letter format"):
        template.render(
            question="Too many?",
            context=None,
            answer_options=answer_options,
            use_option_letters=True,
        )
