""""Neural Network Head Module."""

from typing import Any, Callable

import torch
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim
from torch.optim import lr_scheduler
from typing_extensions import override

from eva.core.metrics import structs as metrics_lib
from eva.core.models.modules import module
from eva.core.models.modules.typings import INPUT_BATCH, MODEL_TYPE
from eva.core.models.modules.utils import batch_postprocess, grad


class HeadModule(module.ModelModule):
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
        postprocess: batch_postprocess.BatchPostProcess | None = None,
    ) -> None:
        """Initializes the neural net head module.

        Args:
            head: The neural network that would be trained on the features.
            criterion: The loss function to use.
            backbone: The feature extractor. If `None`, it will be expected
                that the input batch returns the features directly.
            optimizer: The optimizer to use.
            lr_scheduler: The learning rate scheduler to use.
            metrics: The metric groups to track.
            postprocess: A list of helper functions to apply after the
                loss and before the metrics calculation to the model
                predictions and targets.
        """
        super().__init__(metrics=metrics, postprocess=postprocess)

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
        return self.head(features).squeeze(-1)

    @override
    def on_fit_start(self) -> None:
        if self.backbone is not None:
            grad.deactivate_requires_grad(self.backbone)

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
    def predict_step(self, batch: INPUT_BATCH, *args: Any, **kwargs: Any) -> torch.Tensor:
        tensor = INPUT_BATCH(*batch).data
        return tensor if self.backbone is None else self.backbone(tensor)

    @override
    def on_fit_end(self) -> None:
        if self.backbone is not None:
            grad.activate_requires_grad(self.backbone)

    def _batch_step(self, batch: INPUT_BATCH) -> STEP_OUTPUT:
        """Performs a model forward step and calculates the loss.

        Args:
            batch: The desired batch to process.

        Returns:
            The batch step output.
        """
        data, targets, metadata = INPUT_BATCH(*batch)
        predictions = self(data)
        loss = self.criterion(predictions, targets)
        return {
            "loss": loss,
            "targets": targets,
            "predictions": predictions,
            "metadata": metadata,
        }
