"""G-Eval LLM Judge implementation."""

from typing import Any, List, Sequence, Tuple

from typing_extensions import override

from eva.language.data.messages import UserMessage
from eva.language.metrics.llm_judge import base
from eva.language.metrics.llm_judge.g_eval.template import GEvalPromptTemplate
from eva.language.models import wrappers
from eva.language.models.postprocess import ExtractAnswerFromJson
from eva.language.models.typings import PredictionBatch, TextBatch


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
        model: wrappers.LanguageModel | str | None,
        evaluation_steps: Sequence[str],
        score_range: Tuple[int, int] = (0, 10),
        score_explanation: str = "where higher is better.",
    ):
        """Initializes the G-Eval LLM Judge with a model and prompt template.

        Args:
            model: An instance of the language model to use, or the
                name of the model to load from the registry.
            evaluation_steps: The steps to guide the evaluation.
            score_range: The range of possible scores (min, max).
            score_explanation: Explanation of what the score means.
        """
        super().__init__(model=self._load_model(model), prompt_template=GEvalPromptTemplate())

        self.evaluation_steps = evaluation_steps
        self.score_range = score_range
        self.score_explanation = score_explanation

        self.answer_extractor = ExtractAnswerFromJson(answer_key="score")

    @override
    def evaluate(self, batch: PredictionBatch[List[str]]) -> List[int]:
        """Evaluates a batch of predictions.

        Args:
            batch: A batch of predictions to evaluate against their corresponding targets.

        Returns:
            The evaluation result as a float score.
        """
        prompts = []
        for prediction, target in zip(batch.prediction, batch.target, strict=False):
            prompt = self.prompt_template.render(
                prediction=prediction,
                target=target,
                evaluation_steps=self.evaluation_steps,
                score_range=self.score_range,
                score_explanation=self.score_explanation,
            )

            prompts.append([UserMessage(content=prompt)])

        judge_batch = TextBatch(text=prompts, target=None, metadata=None)

        outputs = self.model(judge_batch)
        results = self.answer_extractor(outputs["generated_text"])

        return list(map(self._cast_to_int, results))

    def _load_model(self, model: wrappers.LanguageModel | str | None) -> wrappers.LanguageModel:
        if model is None or isinstance(model, str):
            return wrappers.ModelFromRegistry(model or self._default_model)  # type: ignore
        return model

    def _cast_to_int(self, result: Any) -> int:
        try:
            return int(result)
        except ValueError as e:
            raise ValueError(f"Score must be an integer, got {result}") from e
