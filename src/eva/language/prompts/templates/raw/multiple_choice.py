"""Prompt templates for multiple choice questions without strict formatting requirements."""

from __future__ import annotations

import string
import textwrap
from typing import Sequence

from jinja2 import Template
from typing_extensions import override

from eva.language.prompts.templates import base


class RawMultipleChoicePromptTemplate(base.PromptTemplate):
    """Prompt template for multiple choice questions only requiring the final answer to be last."""

    template: str = textwrap.dedent(
        """\
        {{ preamble }}

        Question: {{ question }}
        {% if context %}
        Context:
        {{ context }}
        {% endif %}
        Provide a brief explanation for your choice before stating your final answer.

        {%- if enable_cot %}
        Think step-by-step inside <think>...</think> tags before giving your answer.
        {%- endif %}

        IMPORTANT: You must provide your reasoning first.
        Then end your response with only your final choice
        {%- if use_option_letters %} letter
        {%- else %} exactly as written below
        {%- endif %}.
        Do not add any text after that final response.
        {% if use_option_letters %}
        Select the letter (e.g., "A", "B", "C", ...) corresponding to one of the options below:
        {% else %}
        Select exactly one of the options listed below:
        {% endif %}
        {{ answer_options }}

        Example answer:
        {{ example_response }}

        Answer:
        """
    )
    """Base template to be rendered via Jinja2."""

    _default_reason: str = "The reason why the given answer was chosen."

    def __init__(
        self,
        use_option_letters: bool = False,
        enable_cot: bool = False,
    ) -> None:
        """Initializes the prompt template.

        Args:
            use_option_letters: Whether to prefix options with letters (A, B, C, ...).
            enable_cot: Whether to explicitly prompt the model to use reasoning/CoT for answering.
        """
        super().__init__()

        self.use_option_letters = use_option_letters
        self.enable_cot = enable_cot

    @override
    def render(
        self,
        *,
        question: str,
        context: str | Sequence[str] | None,
        answer_options: Sequence[str],
        example_answer: str | None = None,
        example_reason: str | None = None,
        preamble: str | None = None,
    ) -> str:
        """Render the template with provided values.

        Args:
            question: The question to ask the model.
            context: Supporting context text(s) for the question.
            answer_options: Allowed answer options.
            example_answer: Optional example answer. Defaults to first option.
            example_reason: Example reasoning string.
            preamble: Optional preamble text to include at the top of the prompt.

        Returns:
            The rendered prompt string.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("`question` must be a non-empty string.")

        answer_options = _format_answer_options(
            answer_options, use_option_letters=self.use_option_letters
        )
        example_answer = (
            example_answer
            if isinstance(example_answer, str)
            else (string.ascii_uppercase[0] if self.use_option_letters else answer_options[0])
        ).strip()
        example_reason = example_reason or self._default_reason

        jinja_template = Template(self.template)
        rendered = jinja_template.render(
            question=question.strip(),
            context=_format_context(context) if context else None,
            answer_options=answer_options,
            preamble=(preamble or "").strip(),
            use_option_letters=self.use_option_letters,
            enable_cot=self.enable_cot,
            example_response="\n".join([example_reason, example_answer]),
        )

        return textwrap.dedent(rendered).strip() + "\n"


def _format_answer_options(options: Sequence[str], use_option_letters: bool) -> str:
    """Format answer options for inclusion in the prompt.

    Args:
        options: List of answer options.
        use_option_letters: Whether to prefix options with letters (A, B, C, ...).
    """
    if not options or not all(isinstance(opt, str) and opt.strip() for opt in options):
        raise ValueError("`answer_options` must contain at least one non-empty option.")

    if use_option_letters:
        letters = string.ascii_uppercase
        if len(options) > len(letters):
            raise ValueError(f"If using option letters, max {len(letters)} options are supported.")
        return "\n".join(f"{letters[i]}. {opt.strip()}" for i, opt in enumerate(options))
    else:
        return "\n".join(f"- {opt.strip()}" for i, opt in enumerate(options))


def _format_context(context: str | Sequence[str]) -> str:
    """Formats the context for inclusion in the prompt.

    Args:
        context: The context string or list of context strings. If a list is provided,
                 the contexts will be formatted as a bullet point list.

    Returns:
        The formatted context string.
    """
    if not isinstance(context, list):
        context = [context]  # type: ignore[assignment]
    return "\n".join(f"- {item.strip()}" for item in context if item.strip())
