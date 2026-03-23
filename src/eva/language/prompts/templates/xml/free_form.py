"""Prompt templates for freeform questions with XML output."""

# ruff: noqa: E501
from __future__ import annotations

import textwrap
from typing import Sequence

from jinja2 import Template
from typing_extensions import override

from eva.language.prompts.templates import base, typings
from eva.language.utils.text import format as format_utils


class XmlFreeFormQuestionPromptTemplate(base.PromptTemplate):
    """Prompt template for freeform questions while enforcing XML output."""

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

        {% if enable_cot -%}
        IMPORTANT:  Think step-by-step before giving your final answer, then provide your final answer within <{{ answer_key }}></{{ answer_key }}> tags.
        {%- else -%}
        IMPORTANT: Provide your final answer within <{{ answer_key }}></{{ answer_key }}> tags.
        {%- endif %}

        {% if not examples %}
        Example Answer:
        Your explanation for why you chose this answer can go here...
        <{{ answer_key }}>{{ example_answer }}</{{ answer_key }}>
        {% endif %}

        Answer:
        """
    )
    """Base template to be rendered via Jinja2."""

    _default_answer_key: str = "answer"
    """Default key name for the answer in the XML output."""

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
        example_answer: str | None = None,
        preamble: str | None = None,
        enable_cot: bool | None = None,
        answer_key: str | None = None,
    ) -> str:
        """Render the template with provided values.

        Args:
            question: The question to ask the model.
            context: Supporting context text(s) for the question.
            examples: Optional list of example question-answer pairs.
                Each example should be a QuestionAnswerExample with 'question' and 'answer' keys.
            example_answer: Optional example answer for the XML snippet. Defaults to first option.
            preamble: Optional preamble text to include at the top of the prompt.
            enable_cot: Whether to explicitly prompt the model to use reasoning/CoT for answering.
            answer_key: Key name for the answer in the XML output. Defaults to "answer".

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
            answer_key=answer_key or self._default_answer_key,
            preamble=(preamble or "").strip(),
            enable_cot=enable_cot,
        )
        return format_utils.remove_multi_blank_lines(textwrap.dedent(rendered).strip() + "\n")
