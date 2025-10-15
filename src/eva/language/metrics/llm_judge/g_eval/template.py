"""Prompt templates for G-Eval LLM Judge metric."""

# ruff: noqa: E501

from __future__ import annotations

import textwrap
from typing import Sequence, Tuple

from jinja2 import Template
from typing_extensions import override

from eva.language.prompts.templates import base


class GEvalPromptTemplate(base.PromptTemplate):
    """Prompt template for G-Eval LLM Judge metric.

    The template being used here was strongly inspired
    by https://github.com/confident-ai/deepeval.
    """

    template: str = textwrap.dedent(
        """\
        You are an evaluator. Given the following evaluation steps, assess the Model Response below and return a JSON object with two fields:

        - `"score"`: an integer between {{ score_range[0] }} and {{ score_range[1] }}, {{ score_explanation }}.
        - `"reason"`: a brief explanation for why the score was given. This must mention specific strengths or shortcomings, referencing relevant details from the input. Do **not** quote the score itself in the explanation.

        Your explanation should:
        - Be specific and grounded in the evaluation steps.
        - Mention key details from the Model Response and Ground Truth.
        - Be concise, clear, and focused on the evaluation logic.

        Only return valid JSON. Do **not** include any extra commentary or text.

        ---

        Evaluation Steps:
        {{ evaluation_steps }}

        {% if scoring_criteria %}
        Scoring Criteria:
        {{ scoring_criteria }}
        {% endif %}

        Model Response:
        {{ prediction }}

        Ground Truth:
        {{ target }}

        {% if parameters %}
        Parameters:
        {{ parameters }}
        {% endif %}

        {% if additional_context %}
        Additional Context:
        {{ additional_context }}
        {% endif %}

        ---
        **Example JSON:**
        {
            "reason": "your concise and informative reason here",
            "score": {{ score_range[0] }}
        }

        JSON:
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
        prediction: str,
        target: str,
        evaluation_steps: Sequence[str],
        score_range: Tuple[int, int],
        score_explanation: str,
        scoring_criteria: str | None = None,
        additional_context: str | Sequence[str] | None = None,
    ) -> str:
        """Render the template with provided values.

        Args:
            prediction: The model's prediction to evaluate.
            target: The ground truth target to compare against.
            evaluation_steps: The steps to guide the evaluation.
            score_range: The range of possible scores (min, max).
            score_explanation: Explanation of what the score means.
            scoring_criteria: The detailed scoring criteria to include in the prompt.
            additional_context: Any additional context to include in the prompt.

        Returns:
            The rendered prompt string.
        """
        jinja_template = Template(self.template)
        rendered = jinja_template.render(
            score_range=score_range,
            score_explanation=score_explanation,
            scoring_criteria=scoring_criteria,
            evaluation_steps=_format_numbered_list(evaluation_steps),
            prediction=prediction,
            target=target,
            parameters=None,
            additional_context=additional_context,
        )

        # TODO: remove multi blank lines here
        # (wait for https://github.com/kaiko-ai/eva/pull/912 to be merged)
        return textwrap.dedent(rendered).strip() + "\n"


def _format_numbered_list(items: Sequence[str]) -> str:
    """Return items as a numbered bullet point list."""
    # TODO: move this to utils & combine with existing bullet point formatting function
    # (wait for https://github.com/kaiko-ai/eva/pull/912 to be merged)
    formatted = []
    for i, item in enumerate(items, start=1):
        if not isinstance(item, str):
            raise TypeError(f"Expected evaluation step to be a string, got {type(item)}")
        stripped = item.strip()
        if stripped:
            formatted.append(f"{i}. {stripped}")
    return "\n".join(formatted)
