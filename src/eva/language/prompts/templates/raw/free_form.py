"""Prompt templates for free-form questions."""

from __future__ import annotations

import textwrap
from typing import Sequence

from jinja2 import Template
from typing_extensions import override

from eva.language.prompts.templates import base, typings
from eva.language.utils.text import format as format_utils


class FreeFormQuestionPromptTemplate(base.PromptTemplate):
    """Prompt template for free-form questions."""

    template = textwrap.dedent(
        """\
        {{ preamble }}

        {% if examples %}
        Below are some examples:

        {% for ex in examples %}
        Example {{ loop.index }}:
        Question: {{ ex.question }}
        {% if ex.context %}
        Context:
        {{ ex.context }}
        {% endif %}
        Answer: {{ ex.answer }}
        ---
        {% endfor %}
        Now please answer the following question.
        {%- if enable_cot %}
        Think step-by-step inside <think>...</think> tags before giving your answer.
        {% endif %}

        {% endif %}
        Question: {{ question }}
        {% if context %}
        Context:
        {{ context }}
        {% endif %}

        Answer:
        """
    )
    """Base template to be rendered via Jinja2."""

    def __init__(self, enable_cot: bool = False) -> None:
        """Initializes the prompt template.

        Args:
            enable_cot: Whether to explicitly prompt the model to use reasoning/CoT for answering.
        """
        super().__init__()
        self.enable_cot = enable_cot

    @override
    def render(
        self,
        *,
        question: str,
        context: str | Sequence[str] | None = None,
        examples: Sequence[typings.QuestionAnswerExample] | None = None,
        preamble: str | None = None,
        enable_cot: bool | None = None,
    ) -> str:
        """Render the template with provided values.

        Args:
            question: The question to ask the model.
            context: Supporting context text(s) for the question.
            examples: A sequence of question & answer pairs to include as examples.
                Expected format is a list of dicts with 'question', 'answer', and
                optional 'context' keys.
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
            examples=self._validate_and_format_examples(examples),
            preamble=(preamble or "").strip(),
            enable_cot=self.enable_cot if enable_cot is None else enable_cot,
        )

        return format_utils.remove_multi_blank_lines(textwrap.dedent(rendered).strip()) + "\n"

    def _validate_and_format_examples(
        self, examples: Sequence[typings.QuestionAnswerExample] | None
    ) -> Sequence[typings.QuestionAnswerExample] | None:
        """Validates the format of the provided examples."""
        if examples is not None:
            if not isinstance(examples, Sequence):
                raise ValueError(
                    "`examples` must be a sequence of dictionaries, got {type(examples)}."
                )

            for idx, ex in enumerate(examples):
                if not isinstance(ex, dict):
                    raise ValueError(f"Example at index {idx} is not a dictionary.")
                if (
                    "question" not in ex
                    or not isinstance(ex["question"], str)
                    or not ex["question"].strip()
                ):
                    raise ValueError(f"Example at index {idx} is missing a valid 'question' key.")
                if (
                    "answer" not in ex
                    or not isinstance(ex["answer"], str)
                    or not ex["answer"].strip()
                ):
                    raise ValueError(f"Example at index {idx} is missing a valid 'answer' key.")
                if "context" in ex:
                    if not isinstance(ex["context"], str) and not isinstance(
                        ex["context"], Sequence
                    ):
                        raise ValueError(f"Example at index {idx} has an invalid 'context' key.")
                    ex["context"] = format_utils.format_as_bullet_points(ex["context"])
        return examples
