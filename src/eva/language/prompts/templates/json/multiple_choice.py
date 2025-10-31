"""Prompt templates for multiple choice questions with JSON output."""

# ruff: noqa: E501

from __future__ import annotations

import string
import textwrap
from typing import Sequence

from jinja2 import Template
from typing_extensions import override

from eva.language.prompts.templates import base, typings
from eva.language.utils.text import format as format_utils


class JsonMultipleChoicePromptTemplate(base.PromptTemplate):
    """Prompt template for multiple choice questions while enforcing JSON output."""

    template: str = textwrap.dedent(
        """\
        {{ preamble }}

        {% if examples %}
        Below are some examples of how to answer questions:

        {% for ex in examples %}
        Example {{ loop.index }}:
        Question: {{ ex.question }}
        Answer: {{ ex.answer }}
        ---
        {% endfor %}
        Now please answer the following question.
        {% endif %}

        Question: {{ question }}
        {% if context %}
        Context:
        {{ context }}
        {% endif %}

        IMPORTANT: Respond with a valid JSON object where the "{{ answer_key }}" key contains your answer.
        {% if enable_cot -%}
        Think step-by-step before giving your final answer.
        {%- endif -%}
        {% if use_option_letters %}
        The value for "{{ answer_key }}" must be the letter (e.g., "A", "B", "C", ...)
        corresponding to your chosen option from the list below:
        {% else %}
        The value for "{{ answer_key }}" must exactly match one of the options listed below:
        {% endif %}
        {{ answer_options }}


        {% if not examples %}
        Example JSON Answer:
        Your explanation for why you chose this answer can go here...
        {{ '{' }}
            "{{ answer_key }}": "{{ example_answer }}"
        {{ '}' }}
        {% endif %}

        Answer:
        """
    )
    """Base template to be rendered via Jinja2."""

    _default_answer_key: str = "answer"
    """Default key name for the answer in the JSON output."""

    def __init__(
        self,
    ) -> None:
        """Initializes the prompt template."""
        super().__init__()

    @override
    def render(
        self,
        *,
        question: str,
        context: str | Sequence[str] | None,
        answer_options: Sequence[str],
        examples: Sequence[typings.QuestionAnswerExample] | None = None,
        example_answer: str | None = None,
        preamble: str | None = None,
        use_option_letters: bool | None = None,
        enable_cot: bool | None = None,
        answer_key: str | None = None,
    ) -> str:
        """Render the template with provided values.

        Args:
            question: The question to ask the model.
            context: Supporting context text(s) for the question.
            answer_options: Allowed answer options.
            examples: A sequence of question & answer pairs to include as examples.
                Expected format is a list of dicts with 'question', 'answer', and
                optional 'context' keys.
            example_answer: Optional example answer for the JSON snippet. Defaults to first option.
            preamble: Optional preamble text to include at the top of the prompt.
            use_option_letters: Whether to prefix options with letters (A, B, C, ...).
            enable_cot: Whether to explicitly prompt the model to use reasoning/CoT for answering.
            answer_key: Key name for the answer in the JSON output. Defaults to "answer".

        Returns:
            The rendered prompt string.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("`question` must be a non-empty string.")

        jinja_template = Template(self.template)
        rendered = jinja_template.render(
            question=question.strip(),
            context=format_utils.format_list_items(context) if context else None,
            answer_options=format_utils.format_list_items(
                answer_options, style="letters" if use_option_letters else "bullets"
            ),
            answer_key=answer_key or self._default_answer_key,
            examples=examples,
            example_answer=(
                example_answer
                if isinstance(example_answer, str)
                else (string.ascii_uppercase[0] if use_option_letters else answer_options[0])
            ).strip(),
            preamble=(preamble or "").strip(),
            use_option_letters=use_option_letters,
            enable_cot=enable_cot,
        )

        return format_utils.remove_multi_blank_lines(textwrap.dedent(rendered).strip() + "\n")
