"""G-Eval LLM Judge implementation."""

from typing import Any, List, Sequence, Tuple

from loguru import logger
from torch import nn
from typing_extensions import override

from eva.language.data.messages import UserMessage
from eva.language.metrics.llm_judge import base
from eva.language.metrics.llm_judge.g_eval.template import GEvalPromptTemplate
from eva.language.models import wrappers
from eva.language.models.typings import PredictionBatch, TextBatch
from eva.language.utils.text import json as json_utils


class GEvalJudge(base.LLMJudge[int]):
    """G-Eval LLM Judge.

    This is a simplified version of the original G-Eval framework:
    - Evaluation Steps are provided as input to the prompt, rather than
        being produced on-the-fly by the model.
    - No confidence weighted scoring (this requires access log probabilities,
        which is not available for many API models).

    Source: https://arxiv.org/abs/2303.16634
    """

    _default_model = "google/gemini-2.5-flash-lite"

    def __init__(
        self,
        model: wrappers.LanguageModel | nn.Module | str | None,
        evaluation_steps: Sequence[str],
        score_range: Tuple[int, int] = (0, 10),
        score_explanation: str = "where higher is better",
        scoring_criteria: str | None = None,
    ):
        """Initializes the G-Eval LLM Judge with a model and prompt template.

        Args:
            model: An instance of the language model to use, or the
                name of the model to load from the registry.
            evaluation_steps: The steps to guide the evaluation.
            score_range: The range of possible scores (min, max).
            score_explanation: Explanation of what the score means.
            scoring_criteria: The detailed scoring criteria to include in the prompt.
        """
        super().__init__(model=self._load_model(model), prompt_template=GEvalPromptTemplate())

        self.evaluation_steps = evaluation_steps
        self.score_range = score_range
        self.score_explanation = score_explanation
        self.scoring_criteria = scoring_criteria

    @override
    def evaluate(self, batch: PredictionBatch[List[str]]) -> List[int | None]:
        """Evaluates a batch of predictions.

        Args:
            batch: A batch of predictions to evaluate against their corresponding targets.

        Returns:
            The numerical score of the evaluation. Returns None for samples where
            the score could not be extracted.
        """
        prompts = []
        for prediction, target in zip(batch.prediction, batch.target, strict=False):
            prompt = self.prompt_template.render(
                prediction=prediction,
                target=target,
                evaluation_steps=self.evaluation_steps,
                score_range=self.score_range,
                score_explanation=self.score_explanation,
                scoring_criteria=self.scoring_criteria,
            )

            prompts.append([UserMessage(content=prompt)])

        judge_batch = TextBatch(text=prompts, target=None, metadata=None)
        outputs = self.model(judge_batch)
        logger.debug(
            "\n".join(
                [
                    f"prompt:\n{prompt[0].content}\noutput:\n{output}"
                    for prompt, output in zip(prompts, outputs["generated_text"], strict=False)
                ]
            )
        )

        results = list(map(self._parse_output, outputs["generated_text"]))
        scores, reasons = zip(*results, strict=False)

        return list(scores)

    def _load_model(self, model: wrappers.LanguageModel | nn.Module | str | None) -> nn.Module:
        if model is None or isinstance(model, str):
            return wrappers.ModelFromRegistry(model or self._default_model)  # type: ignore
        return model

    def _parse_output(self, output: Any) -> Tuple[int | None, str]:
        json = json_utils.extract_json(output)

        if json is not None:
            if "score" not in json or "reason" not in json:
                raise ValueError("Extracted JSON is missing required 'score' or 'reason' fields.")

            score = int(json["score"])
            if not (self.score_range[0] <= score <= self.score_range[1]):
                raise ValueError(f"Score {score} is out of the expected range {self.score_range}.")

            return score, str(json["reason"])
        else:
            return None, "Failed to extract JSON from model output."
