"""Base class for LLM Judge implementations."""

import abc
from typing import Generic, List, TypeVar

from torch import nn

from eva.language.models import wrappers
from eva.language.models.typings import PredictionBatch
from eva.language.prompts import templates

JudgeOutput = TypeVar("JudgeOutput")


class LLMJudge(Generic[JudgeOutput], abc.ABC):
    """Base class for LLM Judge implementations."""

    def __init__(
        self,
        model: wrappers.LanguageModel | nn.Module,
        prompt_template: templates.PromptTemplate,
    ):
        """Initializes the LLMJudge with a model name and prompt template.

        Args:
            model: An instance of the language model to use, or the
                name of the model to load from the registry.
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
            batch: A batch containing predictions & targets.

        Returns:
            A list of evaluation results, one for each prediction.
        """
        raise NotImplementedError
