"""Base model module."""
from typing import Any, Callable, TypeVar

import torch
from pytorch_lightning.cli import LRSchedulerCallable, OptimizerCallable
from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing_extensions import override

from eva.metrics import core as metrics_lib
from eva.models import _utils, module
from eva.models.typings import ModelType

INPUT_BATCH = TypeVar("INPUT_BATCH")
"""The input batch type."""


class NNHead(module.ModelModule[INPUT_BATCH]):
    """@"""

    def __init__(
        self,
        head: ModelType,
        criterion: Callable[..., torch.Tensor],
        backbone: ModelType | None = None,
        optimizer: OptimizerCallable | None = None,
        lr_scheduler: LRSchedulerCallable | None = None,
        metrics: metrics_lib.MetricsSchema | None = None,
    ) -> None:
        """Initializes the linear classification module.

        Args:
            metrics: The metrics schema. Defaults to `None`.
        """
        super().__init__(metrics=metrics)

        self.head = head
        self.criterion = criterion
        self.backbone = backbone
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    @override
    def configure_optimizers(self) -> Any:
        parameters = list(self.head.parameters())
        optimizer = self.optimizer(parameters)
        lr_scheduler = self.lr_scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    @override
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        features = tensor if self.backbone is None else self.model.forward(tensor)
        return self.head(features.flatten(start_dim=1))

    @override
    def on_fit_start(self) -> None:
        if self.backbone is not None:
            _utils.deactivate_requires_grad(model=self.backbone)

    @override
    def training_step(self, batch: INPUT_BATCH, batch_idx: int) -> STEP_OUTPUT:
        return self._batch_step(batch, batch_idx)

    @override
    def validation_step(
        self, batch: INPUT_BATCH, batch_idx: int, dataloader_idx: int = 0
    ) -> STEP_OUTPUT:
        return self._batch_step(batch, batch_idx)

    @override
    def test_step(self, batch: INPUT_BATCH, batch_idx: int, dataloader_idx: int = 0) -> STEP_OUTPUT:
        return self._batch_step(batch, batch_idx)

    @override
    def on_fit_end(self) -> None:
        if self.backbone is not None:
            _utils.activate_requires_grad(model=self.backbone)

    def _batch_step(
        self,
        batch: INPUT_BATCH,
        batch_idx: int,
    ) -> STEP_OUTPUT:
        """Performs a model forward step and calculates the loss.

        Args:
            batch: The desired batch to process.
            batch_idx: The index of the input batch.

        Returns:
            The batch step output.
        """
        data, targets = batch
        predictions = self.model(data)
        loss = self.criterion(predictions, targets)
        return {
            "loss": loss,
            "targets": targets,
            "predictions": predictions,
        }
