""""Neural Network Head Module."""
from typing import Any, Callable

import torch
from pytorch_lightning.cli import LRSchedulerCallable, OptimizerCallable
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import optim
from torch.optim import lr_scheduler
from typing_extensions import override

from eva.metrics import core as metrics_lib
from eva.models.modules import _utils, module
from eva.models.modules.typings import MODEL_TYPE, TUPLE_INPUT_BATCH

# TODO this will be expanded to support dict as well
INPUT_BATCH = TUPLE_INPUT_BATCH
"""The input batch annotation."""


class HeadModule(module.ModelModule[INPUT_BATCH]):
    """Neural Net Head Module for training on features.

    It can be used for supervised (mini-batch) stochastic gradient descent
    downstream tasks such as classification, regression and segmentation.
    """

    def __init__(
        self,
        head: MODEL_TYPE,
        criterion: Callable[..., torch.Tensor],
        backbone: MODEL_TYPE | None = None,
        optimizer: OptimizerCallable = optim.Adam,
        lr_scheduler: LRSchedulerCallable = lr_scheduler.ConstantLR,
        metrics: metrics_lib.MetricsSchema | None = None,
    ) -> None:
        """Initializes the neural net head module.

        Args:
            head: The neural network that would be trained on the features.
            criterion: The loss function to use.
            backbone: The feature extractor. If `None`, it will be expected
                that the input batch returns the features directly.
                Defaults to `None`.
            optimizer: The optimizer to use.
                Defaults to :class:`torch.optim.Adam`.
            lr_scheduler: The learning rate scheduler to use.
                Defaults to :class:`torch.optim.lr_scheduler.ConstantLR`.
            metrics: The list of metrics to track. If `None`, it uses
                the :meth:`self.default_metrics`. Defaults to `None`.
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
    def forward(self, tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        features = tensor if self.backbone is None else self.backbone(tensor)
        return self.head(features.flatten(start_dim=1))

    @override
    def on_fit_start(self) -> None:
        if self.backbone is not None:
            _utils.deactivate_requires_grad(self.backbone)

    @override
    def training_step(self, batch: INPUT_BATCH, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self._batch_step(batch)

    @override
    def validation_step(self, batch: INPUT_BATCH, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self._batch_step(batch)

    @override
    def test_step(self, batch: INPUT_BATCH, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self._batch_step(batch)

    @override
    def on_fit_end(self) -> None:
        if self.backbone is not None:
            _utils.activate_requires_grad(self.backbone)

    def _batch_step(self, batch: INPUT_BATCH) -> STEP_OUTPUT:
        """Performs a model forward step and calculates the loss.

        Args:
            batch: The desired batch to process.

        Returns:
            The batch step output.
        """
        data, targets = batch[0], batch[1]
        predictions = self(data)
        loss = self.criterion(predictions, targets)
        return {
            "loss": loss,
            "targets": targets,
            "predictions": predictions,
        }
