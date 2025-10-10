"""Base class for LLM Judge implementations."""

import abc
from typing import Generic, List, TypeVar

from eva.language.models import wrappers
from eva.language.models.typings import PredictionBatch
from eva.language.prompts import templates

JudgeOutput = TypeVar("JudgeOutput")


class LLMJudge(Generic[JudgeOutput], abc.ABC):
    """Base class for LLM Judge implementations."""

    def __init__(
        self,
        model: wrappers.LanguageModel,
        prompt_template: templates.PromptTemplate,
    ):
        """Initializes the LLMJudge with a model name and prompt template.

        Args:
            model_name: The name of the model to use for evaluation. It requires prepending
                the API provider such as "gemini/gemini-2.5-flash-lite" for Gemini.
            api_kwargs: Additional keyword arguments for the API.
            prompt_template: The template to use for the prompt.
        """
        self.model = model
        self.prompt_template = prompt_template

    @abc.abstractmethod
    def evaluate(
        self,
        batch: PredictionBatch[str],
    ) -> List[JudgeOutput]:
        """Evaluates a batch of predictions.

        Args:
            predictions: A list of model predictions to evaluate.
            targets: A list of ground truth targets to compare against (optional).
            contexts: A list of additional contexts to consider during evaluation (optional).

        Returns:
            A list of evaluation results, one for each prediction.
        """
        raise NotImplementedError
