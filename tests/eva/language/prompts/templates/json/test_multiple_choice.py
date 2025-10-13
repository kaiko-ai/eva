"""Tests for JSON multiple choice prompt template."""

import pytest

from eva.language.prompts.templates.json.multiple_choice import JsonMultipleChoicePromptTemplate


@pytest.fixture
def template() -> JsonMultipleChoicePromptTemplate:
    """Return a baseline template instance for reuse across tests."""
    return JsonMultipleChoicePromptTemplate()


def test_render_basic_trims_input(template: JsonMultipleChoicePromptTemplate) -> None:
    """Renders the core prompt with minimal inputs and trims whitespace."""
    result = template.render(
        question="  What is the capital of France?  ",
        context=None,
        answer_options=["  Paris  ", "London"],
    )

    assert result.startswith("Question: What is the capital of France?")
    assert "- Paris" in result
    assert "- London" in result
    assert '"answer":' in result and '"reason":' in result


def test_render_context_formats_lists(template: JsonMultipleChoicePromptTemplate) -> None:
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
        (True, "A. Red", 'The value for "answer" must be the letter'),
        (False, "- Red", 'The value for "answer" must exactly match'),
    ],
)
def test_render_option_styles(
    use_letters: bool, expected_option: str, instruction_snippet: str
) -> None:
    """Rendering switches between lettered options and bullet options."""
    template = JsonMultipleChoicePromptTemplate(use_option_letters=use_letters)
    result = template.render(
        question="Pick a color",
        context=None,
        answer_options=["Red", "Blue"],
    )

    assert expected_option in result
    assert instruction_snippet in result


@pytest.mark.parametrize(
    ("answer_key", "reason_key"),
    [
        ("choice", "why"),
        ("response", "rationale"),
    ],
)
def test_render_custom_keys_respected(answer_key: str, reason_key: str) -> None:
    """Custom answer_key/reason_key propagate to the instructions and example JSON."""
    template = JsonMultipleChoicePromptTemplate(answer_key=answer_key, reason_key=reason_key)
    result = template.render(
        question="Test question",
        context=None,
        answer_options=["A", "B"],
    )

    assert f'The value for "{answer_key}"' in result
    assert f'"{answer_key}":' in result
    assert f'"{reason_key}":' in result
    assert '"answer":' not in result
    assert '"reason":' not in result


@pytest.mark.parametrize(
    ("use_letters", "example_answer", "expected_fragment"),
    [
        (False, None, '"answer": "First"'),
        (True, None, '"answer": "A"'),
        (False, "  Second  ", '"answer": "Second"'),
    ],
)
def test_render_example_answer_selection(
    use_letters: bool, example_answer: str | None, expected_fragment: str
) -> None:
    """Default and user-supplied example answers should match expectations."""
    template = JsonMultipleChoicePromptTemplate(use_option_letters=use_letters)
    kwargs = {"example_answer": example_answer} if example_answer is not None else {}
    result = template.render(
        question="Example answer?",
        context=None,
        answer_options=["First", "Second"],
        enable_cot=False,
        **kwargs,
    )

    assert expected_fragment in result


@pytest.mark.parametrize(
    ("enable_cot", "expected_fragment"),
    [
        (False, "<think>"),
        (True, "<think>"),
    ],
)
def test_render_enable_cot(enable_cot: bool, expected_fragment: str) -> None:
    """Prompt with enable_cot should contain a fragment asking the model to use thinking/CoT."""
    template = JsonMultipleChoicePromptTemplate(enable_cot=enable_cot)
    result = template.render(
        question="Example answer?",
        context=None,
        answer_options=["First", "Second"],
    )
    if enable_cot:
        assert expected_fragment in result
    else:
        assert expected_fragment not in result


@pytest.mark.parametrize("question", ["", "   ", 123])
def test_render_invalid_question_raises_error(
    template: JsonMultipleChoicePromptTemplate, question: object
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
    template: JsonMultipleChoicePromptTemplate, answer_options: list[object]
) -> None:
    """Invalid answer options should raise a descriptive ValueError."""
    with pytest.raises(ValueError, match="`answer_options` must contain at least one"):
        template.render(
            question="Options?",
            context=None,
            answer_options=answer_options,  # type: ignore[arg-type]
        )


def test_render_with_too_many_lettered_options() -> None:
    """Using option letters should enforce the alphabet upper bound."""
    template = JsonMultipleChoicePromptTemplate(use_option_letters=True)
    answer_options = [f"Option {i}" for i in range(27)]

    with pytest.raises(ValueError, match="max 26 options are supported"):
        template.render(
            question="Too many?",
            context=None,
            answer_options=answer_options,
        )
