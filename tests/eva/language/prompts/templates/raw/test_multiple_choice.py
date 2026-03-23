"""Tests for raw multiple choice prompt template."""

import pytest

from eva.language.prompts.templates.raw.multiple_choice import RawMultipleChoicePromptTemplate


@pytest.fixture
def template() -> RawMultipleChoicePromptTemplate:
    """Return a baseline template instance for reuse across tests."""
    return RawMultipleChoicePromptTemplate()


def test_raw_render_basic_trims_input(template: RawMultipleChoicePromptTemplate) -> None:
    """Renders the core prompt with minimal inputs and trims whitespace."""
    result = template.render(
        question="  What is the capital of France?\t\r ",
        context=None,
        answer_options=["  Paris  ", "London  ", "\tBerlin\n"],
    )

    assert result.startswith("Question: What is the capital of France?")
    assert "- Paris" in result
    assert "- London" in result


def test_raw_render_context_formats_lists(template: RawMultipleChoicePromptTemplate) -> None:
    """Context lists should be bullet formatted and ignore blank entries."""
    result = template.render(
        question="Context handling?",
        context=["First fact", "   Second fact   "],
        answer_options=["Yes", "No"],
    )

    # Extract the context block to confirm only non-empty entries remain.
    context_section = result.split("Context:\n", 1)[1].split("\n\n", 1)[0]
    context_lines = [line for line in context_section.splitlines() if line.startswith("- ")]
    assert context_lines == ["- First fact", "- Second fact"]

    result_no_context = template.render(
        question="Skip context?",
        context=None,
        answer_options=["Yes", "No"],
    )
    assert "Context:" not in result_no_context


@pytest.mark.parametrize(
    ("use_letters", "expected_option", "instruction_snippet"),
    [
        (True, "A. Red", "your final answer letter"),
        (False, "- Red", "your final answer exactly as written below"),
    ],
)
def test_raw_render_option_styles(
    use_letters: bool, expected_option: str, instruction_snippet: str
) -> None:
    """Rendering switches between lettered options and bullet options."""
    raw_template = RawMultipleChoicePromptTemplate()
    result = raw_template.render(
        question="Pick a color",
        context=None,
        answer_options=["Red", "Blue"],
        use_option_letters=use_letters,
    )

    assert expected_option in result
    assert instruction_snippet in result


@pytest.mark.parametrize(
    ("use_letters", "example_answer", "expected_fragment"),
    [
        (False, None, "First"),
        (True, None, "A"),
        (False, "  Second  ", "Second"),
    ],
)
def test_raw_render_example_answer_selection(
    use_letters: bool, example_answer: str | None, expected_fragment: str
) -> None:
    """Default and user-supplied example answers should match expectations."""
    raw_template = RawMultipleChoicePromptTemplate()
    result = raw_template.render(
        question="Example answer?",
        context=None,
        answer_options=["First", "Second"],
        enable_cot=False,
        use_option_letters=use_letters,
        example_answer=example_answer,
    )

    assert expected_fragment in result


@pytest.mark.parametrize(
    ("enable_cot", "expected_fragment"),
    [
        (False, "Think step-by-step"),
        (True, "Think step-by-step"),
    ],
)
def test_raw_render_enable_cot(enable_cot: bool, expected_fragment: str) -> None:
    """Prompt with enable_cot should contain a fragment asking the model to use thinking/CoT."""
    raw_render = RawMultipleChoicePromptTemplate()
    result = raw_render.render(
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
def test_raw_render_invalid_question_raises_error(
    template: RawMultipleChoicePromptTemplate, question: object
) -> None:
    """Invalid questions should raise a descriptive ValueError."""
    with pytest.raises(ValueError, match="`question` must be a non-empty string"):
        template.render(
            question=question,  # type: ignore[arg-type]
            context=None,
            answer_options=["A", "B"],
        )


@pytest.mark.parametrize("question", ["", "   ", 123])
def test_render_invalid_question_raises_error(
    template: RawMultipleChoicePromptTemplate, question: object
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
    template: RawMultipleChoicePromptTemplate, answer_options: list[object]
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
    template = RawMultipleChoicePromptTemplate()
    answer_options = [f"Option {i}" for i in range(27)]

    with pytest.raises(ValueError, match="Maximum 26 items supported for letter format"):
        template.render(
            question="Too many?",
            context=None,
            answer_options=answer_options,
            use_option_letters=True,
        )
