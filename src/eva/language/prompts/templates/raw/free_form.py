"""Prompt templates for free-form questions."""

# ruff: noqa: E501

from __future__ import annotations

import textwrap
from typing import Sequence

from jinja2 import Template
from typing_extensions import override

from eva.language.prompts.templates import base, typings
from eva.language.utils.text import format as format_utils


class RawFreeFormQuestionPromptTemplate(base.PromptTemplate):
    """Prompt template for free-form questions."""

    template = textwrap.dedent(
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

        {%- if enable_cot %}
        IMPORTANT: Think step-by-step before giving your final answer. Provide your reasoning first, then end your response with your final answer.
        {%- else -%}
        IMPORTANT: Respond in free form text, and make sure that your final answer is the last part of your response.
        {%- endif %}

        {% if not examples %}
        {% if example_answer %}
        Example Answer:
        {% if enable_cot %}
        First, provide your reasoning for why you chose this answer here...
        Then, provide your final answer: {{ example_answer }}
        {% else %}
        {{ example_answer }}
        {% endif %}
        {% endif %}
        {% endif %}

        Answer:
        """
    )
    """Base template to be rendered via Jinja2."""

    def __init__(self) -> None:
        """Initializes the prompt template."""
        super().__init__()

    @override
    def render(
        self,
        *,
        question: str,
        context: str | Sequence[str] | None = None,
        examples: Sequence[typings.QuestionAnswerExample] | None = None,
        example_answer: str | None = None,
        preamble: str | None = None,
        enable_cot: bool | None = None,
    ) -> str:
        """Render the template with provided values.

        Args:
            question: The question to ask the model.
            context: Supporting context text(s) for the question.
            examples: Optional list of example question-answer pairs.
                Each example should be a dict with 'question' and 'answer' keys.
            example_answer: Optional example answer for the raw snippet. Defaults to first option.
            preamble: Optional preamble text to include at the top of the prompt.
            enable_cot: Optionally override the instance's CoT setting for this render call.

        Returns:
            The rendered prompt string.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("`question` must be a non-empty string.")

        if isinstance(context, Sequence) and not isinstance(context, str):
            context = format_utils.format_list_items(context)

        jinja_template = Template(self.template)
        rendered = jinja_template.render(
            question=question.strip(),
            context=context if context else None,
            examples=examples,
            example_answer=example_answer,
            preamble=(preamble or "").strip(),
            enable_cot=enable_cot if enable_cot is None else enable_cot,
        )

        return format_utils.remove_multi_blank_lines(textwrap.dedent(rendered).strip() + "\n")
