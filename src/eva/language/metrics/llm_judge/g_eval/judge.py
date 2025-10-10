from typing import List, Sequence, Tuple

from typing_extensions import override

from eva.language.data.messages import UserMessage
from eva.language.metrics.llm_judge import base
from eva.language.metrics.llm_judge.g_eval.template import GEvalPromptTemplate
from eva.language.models import wrappers
from eva.language.models.postprocess import ExtractAnswerFromJson
from eva.language.models.typings import PredictionBatch, TextBatch


class GEvalJudge(base.LLMJudge[float]):
    def __init__(
        self,
        model: wrappers.LanguageModel,
        evaluation_steps: Sequence[str],
        score_range: Tuple[int, int] = (0, 10),
    ):
        """Initializes the G-Eval LLM Judge with a model and prompt template.

        Args:
            model: The language model to use for evaluation.
            prompt_template: The template to use for the prompt.
        """
        super().__init__(model=model, prompt_template=GEvalPromptTemplate())

        self.evaluation_steps = evaluation_steps
        self.score_range = score_range

        self.answer_extractor = ExtractAnswerFromJson(answer_key="score")

    @override
    def evaluate(self, batch: PredictionBatch[List[str]]) -> List[float]:
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
                rubric=None,
            )

            prompts.append([UserMessage(content=prompt)])

        judge_batch = TextBatch(text=prompts, target=None, metadata=None)

        outputs = self.model(judge_batch)
        scores = self.answer_extractor(outputs["generated_text"])

        # TODO: Parse the score from the model output
        # return [float(score) for score in scores]
        return scores


if __name__ == "__main__":
    from eva.language.models import wrappers

    model = wrappers.ModelFromRegistry("google/gemini-2.5-flash-lite")

    test_input = TextBatch(
        text=[[UserMessage(content="What is the capital of France?")]],
        target=[None],
        metadata=None,
    )

    output = model(test_input)
    print(output)

    judge = GEvalJudge(
        model=model,
        evaluation_steps=[
            "Correctness: Is the answer factually accurate?",
            "Completeness: Does the answer fully address the question?",
            "Conciseness: Is the answer clear and to the point?",
        ],
        score_range=(0, 10),
    )

    judge_input = PredictionBatch(
        prediction=["The capital of France is Paris."],
        target=["Paris is the capital city of France."],
        metadata=None,
        text=None,
    )

    scores = judge.evaluate(judge_input)
    print(scores)
