"""LLM Text Module for Inference."""

from typing import Any, Dict, Callable
import torch
from torch import nn
from typing_extensions import override
from eva.core.metrics import structs as metrics_lib
from eva.core.models.modules import module
from eva.core.models.modules.typings import INPUT_BATCH
from lightning.pytorch.utilities.types import STEP_OUTPUT
from eva.core.models.modules.utils import batch_postprocess
from eva.core.models.wrappers.huggingface import HuggingFaceModel


class TextModule(module.ModelModule):
    """Text-based LLM module for inference.

    Uses a Hugging Face wrapper for text generation.
    Supports evaluation using configurable metrics and post-processing.
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
            model: A HuggingFace wrapper or another PyTorch-compatible model for text generation.
            metrics: Metrics schema for evaluation.
            postprocess: A helper function to post-process model outputs before evaluation.
        """
        super().__init__(metrics=metrics, postprocess=postprocess)

        self.model = model
        self.prompt = prompt

    @override
    def forward(self, prompts: str, *args: Any, **kwargs: Any) -> list[str]:
        """Generates text responses for a batch of prompts.

        Args:
            prompts: List of input texts to generate responses.

        Returns:
            List of generated responses.
        """
        return self.model.generate(prompts)

    @override
    def validation_step(self, batch: INPUT_BATCH, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        """Validation step that runs batch inference and evaluates metrics."""
        return self._batch_step(batch)

    def _batch_step(self, batch: INPUT_BATCH) -> STEP_OUTPUT:
        """Runs inference on a batch and evaluates model predictions.

        Args:
            batch: A batch of input prompts and ground truth labels.

        Returns:
            Dictionary with predictions, ground truth, and evaluation metrics.
        """
        data, targets, metadata = INPUT_BATCH(*batch)
        message = self.prompt + data + '\nAnswer: '
        predictions = self.forward(message)

        return {
            "predictions": predictions,
            "targets": targets,
            "metadata": metadata,
        }
