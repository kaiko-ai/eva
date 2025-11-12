"""Prompt templates for free-form questions with delimiter-based output format."""

# ruff: noqa: E501

from __future__ import annotations

import textwrap
from typing import Sequence

from jinja2 import Template
from typing_extensions import override

from eva.language.prompts.templates import base, typings
from eva.language.utils.text import format as format_utils


class DelimiterFreeFormQuestionPromptTemplate(base.PromptTemplate):
    """Prompt template for free-form questions with answers after #### delimiter."""

    template: str = textwrap.dedent(
        """\
        {{ preamble }}

        {% if examples %}
        Below are some examples of how to answer questions:

        {% for ex in examples %}
        Example {{ loop.index }}:
        Question: {{ ex.question }}
        {% if ex.context %}
        Context: {{ ex.context }}
        {% endif %}
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

        IMPORTANT: Think step-by-step before giving your final answer, then provide your final answer after "#### ".

        {% if not examples %}
        Example Answer:
        Your explanation and reasoning can go here...
        #### [your final answer here]
        {% endif %}
        """
    )
    """Base template to be rendered via Jinja2."""

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
        context: str | Sequence[str] | None = None,
        examples: Sequence[typings.QuestionAnswerExample] | None = None,
        preamble: str | None = None,
    ) -> str:
        """Render the template with provided values.

        Args:
            question: The question to ask the model.
            context: Supporting context text(s) for the question.
            examples: A sequence of question & answer pairs to include as examples.
                Expected format is a list of dicts with 'question', 'answer', and
                optional 'context' keys.
            preamble: Optional preamble text to include at the top of the prompt.

        Returns:
            The rendered prompt string.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("`question` must be a non-empty string.")

        jinja_template = Template(self.template)
        rendered = jinja_template.render(
            question=question.strip(),
            context=format_utils.format_list_items(context) if context else None,
            examples=examples,
            preamble=(preamble or "").strip(),
        )

        return format_utils.remove_multi_blank_lines(textwrap.dedent(rendered).strip() + "\n")
