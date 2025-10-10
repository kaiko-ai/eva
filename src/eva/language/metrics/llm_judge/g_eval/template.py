"""Prompt templates for G-Eval LLM Judge metric."""

from __future__ import annotations

import textwrap
from typing import Sequence, Tuple

from jinja2 import Template
from typing_extensions import override

from eva.language.prompts.templates import base


class GEvalPromptTemplate(base.PromptTemplate):
    """Prompt template for G-Eval LLM Judge metric."""

    template: str = textwrap.dedent(
        """You are an evaluator. Given the following {{ dependencies }}, assess the response below and return a JSON object with two fields:

        - `"score"`: an integer between {{ score_range[0] }} and {{ score_range[1] }}, {{ score_explanation }}.
        - `"reason"`: a brief explanation for why the score was given. This must mention specific strengths or shortcomings, referencing relevant details from the input. Do **not** quote the score itself in the explanation.

        Your explanation should:
        - {{ reasoning_expectation }}
        - Mention key details from the test case parameters.
        - Be concise, clear, and focused on the evaluation logic.

        Only return valid JSON. Do **not** include any extra commentary or text.

        ---

        Evaluation Steps:
        {{ evaluation_steps }}

        {{ rubric_text }}
        Model Prediction:
        {{ prediction }}

        Ground Truth:
        {{ target }}

        Parameters:
        {{ parameters }}
        {{ additional_context }}

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
        rubric: str | None = None,
        additional_context: str | Sequence[str] | None = None,
    ) -> str:
        """Render the template with provided values.

        Args:

        Returns:
            The rendered prompt string.
        """
        rubric_text = f"Rubric:\n{rubric}\n" if rubric else ""
        dependencies = "evaluation steps and rubric" if rubric else "evaluation steps"
        score_explanation = (
            "based on the rubric provided"
            if rubric
            else f"with {score_range[1]} indicating strong alignment with the evaluation steps and {score_range[0]} indicating no alignment"
        )
        reasoning_expectation = (
            "Be specific and grounded in the evaluation steps and rubric."
            if rubric
            else "Be specific and grounded in the evaluation steps."
        )

        jinja_template = Template(self.template)
        rendered = jinja_template.render(
            dependencies=dependencies,
            score_range=score_range,
            score_explanation=score_explanation,
            reasoning_expectation=reasoning_expectation,
            evaluation_steps="\n".join(
                f"- {step.strip()}" for step in evaluation_steps
            ),  # TODO: use bullet point helper instead with numbers
            rubric_text=rubric_text,
            prediction=prediction,
            target=target,
            parameters=None,
            additional_context=additional_context,
        )

        # TODO: remove multi blank lines here
        return textwrap.dedent(rendered).strip() + "\n"


if __name__ == "__main__":
    template = GEvalPromptTemplate()
    prompt = template.render(
        prediction="The capital of France is Paris.",
        target="Paris is the capital city of France.",
        evaluation_steps=[
            "Check if the answer correctly identifies the capital city of France.",
            "Verify that the answer is concise and directly addresses the question.",
            "Ensure that the answer does not include any irrelevant information.",
        ],
        score_range=(0, 10),
        rubric=None,
        additional_context=None,
    )
    print(prompt)
