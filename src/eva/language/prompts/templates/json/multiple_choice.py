"""Prompt templates for multiple choice questions with JSON output."""

from __future__ import annotations

import string
import textwrap
from typing import Sequence

from jinja2 import Template
from typing_extensions import override

from eva.language.prompts.templates import base
from eva.language.utils.text import format as format_utils


class JsonMultipleChoicePromptTemplate(base.PromptTemplate):
    """Prompt template for multiple choice questions while enforcing JSON output."""

    template: str = textwrap.dedent(
        """\
        {{ preamble }}

        Question: {{ question }}
        {% if context %}
        Context:
        {{ context }}
        {% endif %}

        IMPORTANT: Respond with a valid JSON object where the "{{ answer_key }}" key
        contains your answer, and "{{ reason_key }}" should contain a brief
        explanation for why the provided answer was chosen. 
        {% if enable_cot -%}
        Think step-by-step inside <think>...</think> tags before giving your final answer.
        {%- endif -%}
        {% if use_option_letters %}
        The value for "{{ answer_key }}" must be the letter (e.g., "A", "B", "C", ...)
        corresponding to your chosen option from the list below:
        {% else %}
        The value for "{{ answer_key }}" must exactly match one of the options listed below:
        {% endif %}
        {{ answer_options }}

        Example JSON Answer:
        {{ '{' }}
            "{{ answer_key }}": "{{ example_answer }}",
            "{{ reason_key }}": "{{ example_reason }}"
        {{ '}' }}

        Answer:
        """
    )
    """Base template to be rendered via Jinja2."""

    _default_answer_key: str = "answer"
    """Default key name for the answer in the JSON output."""

    _default_reason_key: str = "reason"
    """Default key name for the reasoning in the JSON output."""

    _default_reason: str = "The reason why the given answer was chosen."
    """Default reasoning string for the example JSON."""

    def __init__(
        self,
        answer_key: str | None = None,
        reason_key: str | None = None,
        use_option_letters: bool = False,
        enable_cot: bool = False,
    ) -> None:
        """Initializes the prompt template.

        Args:
            answer_key: Key name for the answer in the JSON output. Defaults to "answer".
            reason_key: Key name for the reasoning in the JSON output. Defaults to "reason".
            example_reason: Default reasoning string to provide in the `answer_key` for
                the example JSON object in the prompt.
            template: Custom template string to use instead of the default.
            use_option_letters: Whether to prefix options with letters (A, B, C, ...).
            enable_cot: Whether to explicitly prompt the model to use reasoning/CoT for answering.
        """
        super().__init__()

        self.answer_key = answer_key or self._default_answer_key
        self.reason_key = reason_key or self._default_reason_key
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
        enable_cot: bool | None = None,
    ) -> str:
        """Render the template with provided values.

        Args:
            question: The question to ask the model.
            context: Supporting context text(s) for the question.
            answer_options: Allowed answer options.
            example_answer: Optional example answer for the JSON snippet. Defaults to first option.
            example_reason: Example reasoning string.
            preamble: Optional preamble text to include at the top of the prompt.
            enable_cot: Optionally override the instance's CoT setting for this render call.

        Returns:
            The rendered prompt string.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("`question` must be a non-empty string.")

        jinja_template = Template(self.template)
        rendered = jinja_template.render(
            question=question.strip(),
            context=format_utils.format_as_bullet_points(context) if context else None,
            answer_options=_format_answer_options(
                answer_options, use_option_letters=self.use_option_letters
            ),
            answer_key=self.answer_key,
            reason_key=self.reason_key,
            example_answer=(
                example_answer
                if isinstance(example_answer, str)
                else (string.ascii_uppercase[0] if self.use_option_letters else answer_options[0])
            ).strip(),
            example_reason=(example_reason or self._default_reason).strip(),
            preamble=(preamble or "").strip(),
            use_option_letters=self.use_option_letters,
            enable_cot=self.enable_cot if enable_cot is None else enable_cot,
        )

        return format_utils.remove_multi_blank_lines(textwrap.dedent(rendered).strip()) + "\n"


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
