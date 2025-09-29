"""Model module for vision-language models."""

from typing import Any

from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from typing_extensions import override

from eva.core.metrics import structs as metrics_lib
from eva.core.models.modules import module
from eva.core.models.modules.utils import batch_postprocess
from eva.language.models.typings import ModelOutput
from eva.multimodal.models.typings import TextImageBatch


class VisionLanguageModule(module.ModelModule):
    """Model module for vision-language tasks."""

    def __init__(
        self,
        model: nn.Module,
        metrics: metrics_lib.MetricsSchema | None = None,
        postprocess: batch_postprocess.BatchPostProcess | None = None,
    ) -> None:
        """Initializes the text inference module.

        Args:
            model: Model instance to use for forward pass.
            metrics: Metrics schema for evaluation.
            postprocess: A helper function to post-process model outputs before evaluation.
        """
        super().__init__(metrics=metrics, postprocess=postprocess)

        self.model = model

    @override
    def forward(self, batch: TextImageBatch, *args: Any, **kwargs: Any) -> ModelOutput:
        return self.model(batch)

    @override
    def validation_step(self, batch: TextImageBatch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self._batch_step(batch)

    @override
    def test_step(self, batch: TextImageBatch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self._batch_step(batch)

    def _batch_step(self, batch: TextImageBatch) -> STEP_OUTPUT:
        text, _, targets, metadata = TextImageBatch(*batch)
        output = self.forward(batch)
        return {
            "inputs": text,
            "predictions": output.pop("generated_text"),  # type: ignore
            "targets": targets,
            "metadata": metadata,
        } | output
