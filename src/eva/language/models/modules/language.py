"""Model module for language models."""

from typing import Any

from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from typing_extensions import override

from eva.core.metrics import structs as metrics_lib
from eva.core.models.modules import module
from eva.core.models.modules.utils import batch_postprocess
from eva.language.models.typings import ModelOutput, PredictionBatch, TextBatch


class LanguageModule(module.ModelModule):
    """Model module for language tasks."""

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
    def forward(self, batch: TextBatch, *args: Any, **kwargs: Any) -> ModelOutput:
        return self.model(batch)

    @override
    def validation_step(self, batch: TextBatch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self._batch_step(batch)

    @override
    def test_step(self, batch: TextBatch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self._batch_step(batch)

    def _batch_step(self, batch: TextBatch) -> STEP_OUTPUT:
        text, targets, metadata = TextBatch(*batch)
        output = self.forward(batch)

        return {
            "inputs": text,
            "predictions": output.pop("generated_text"),  # type: ignore
            "targets": targets,
            "metadata": metadata,
        } | output


class OfflineLanguageModule(module.ModelModule):
    """Model module for offline language tasks."""

    def __init__(
        self,
        metrics: metrics_lib.MetricsSchema | None = None,
        postprocess: batch_postprocess.BatchPostProcess | None = None,
    ) -> None:
        """Initializes the text inference module.

        Args:
            metrics: Metrics schema for evaluation.
            postprocess: A helper function to post-process model outputs before evaluation.
        """
        super().__init__(metrics=metrics, postprocess=postprocess)

    @override
    def forward(self, batch: PredictionBatch, *args: Any, **kwargs: Any) -> PredictionBatch:
        return batch

    @override
    def validation_step(self, batch: PredictionBatch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self._batch_step(batch)

    @override
    def test_step(self, batch: PredictionBatch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self._batch_step(batch)

    def _batch_step(self, batch: PredictionBatch) -> STEP_OUTPUT:
        predictions, targets, text, metadata = PredictionBatch(*batch)
        return {
            "inputs": text,
            "predictions": predictions,
            "targets": targets,
            "metadata": metadata,
        }
