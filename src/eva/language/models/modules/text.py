"""LLM Text Module for Inference."""

from typing import Any, List

from lightning.pytorch.utilities.types import STEP_OUTPUT
from loguru import logger
from torch import nn
from typing_extensions import override

from eva.core.metrics import structs as metrics_lib
from eva.core.models.modules import module
from eva.core.models.modules.utils import batch_postprocess
from eva.language.models.modules.typings import TEXT_BATCH


class TextModule(module.ModelModule):
    """Text-based LLM module for inference.

    Uses LLM wrappers for text generation and supports evaluation using
    configurable metrics and post-processing transforms.
    """

    def __init__(
        self,
        model: nn.Module,
        prompt: str,
        metrics: metrics_lib.MetricsSchema | None = None,
        postprocess: batch_postprocess.BatchPostProcess | None = None,
    ) -> None:
        """Initializes the text inference module.

        Args:
            model: An LLM wrapper (PyTorch-compatible) for text generation.
            prompt: The prompt to use for generating text.
            metrics: Metrics schema for evaluation.
            postprocess: A helper function to post-process model outputs before evaluation.
        """
        super().__init__(metrics=metrics, postprocess=postprocess)

        self.model = model
        self.prompt = prompt

    @override
    def forward(self, prompts: List[str], *args: Any, **kwargs: Any) -> List[str]:
        """Generates text responses for a batch of prompts.

        Args:
            prompts: List of input texts to generate responses.
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            List of generated responses.
        """
        return self.model(prompts)

    @override
    def validation_step(self, batch: TEXT_BATCH, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        """Validation step that runs batch inference and evaluates metrics.

        Args:
            batch: An input batch.
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            Dictionary with predictions, ground truth, and evaluation metrics.
        """
        return self._batch_step(batch)

    def _batch_step(self, batch: TEXT_BATCH) -> STEP_OUTPUT:
        """Runs inference on a batch and evaluates model predictions.

        Args:
            batch: Input batch containing data, targets, and metadata.

        Returns:
            Dictionary with predictions, ground truth, and evaluation metrics.
        """
        data, targets, metadata = batch
        messages = [str(d) + "\n" + self.prompt for d in data]
        predictions = self(messages)
        logger.debug(f"Predictions: {predictions}")
        logger.debug(f"Targets: {targets}")
        return {"predictions": predictions, "targets": targets, "metadata": metadata}
