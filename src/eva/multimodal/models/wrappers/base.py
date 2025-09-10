"""Base class for vision language model wrappers."""

import abc
from typing import Any, Callable

from typing_extensions import override

from eva.core.models.wrappers import base
from eva.language.data.messages import ModelSystemMessage
from eva.language.models.typings import ModelOutput
from eva.multimodal.models.typings import TextImageBatch


class VisionLanguageModel(base.BaseModel[TextImageBatch, ModelOutput]):
    """Base class for multimodal models.

    Classes that inherit from this should implement the following methods:
    - `load_model`: Loads & instantiates the model.
    - `model_forward`: Implements the forward pass of the model. For API models,
        this can be an API call.
    - `format_inputs`: Preprocesses and converts the input batch into the format
        expected by the `model_forward` method.
    """

    def __init__(
        self, system_prompt: str | None, output_transforms: Callable | None = None
    ) -> None:
        """Creates a new model instance.

        Args:
            system_prompt: The system prompt to use for the model (optional).
            output_transforms: Optional transforms to apply to the output of
                the model's forward pass.
        """
        super().__init__(transforms=output_transforms)

        self.system_message = ModelSystemMessage(content=system_prompt) if system_prompt else None

    @override
    def forward(self, batch: TextImageBatch) -> ModelOutput:
        """Forward pass of the model."""
        inputs = self.format_inputs(batch)
        return super().forward(inputs)

    @abc.abstractmethod
    def format_inputs(self, batch: TextImageBatch) -> Any:
        """Converts the inputs into the format expected by the model."""
        raise NotImplementedError
