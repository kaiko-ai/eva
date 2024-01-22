"""Base model module."""
from typing import Any, Dict, TypeVar, Callable

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import memory
from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing_extensions import override
from torch import nn 
from pytorch_lightning.cli import (
    LRSchedulerCallable,
    OptimizerCallable,
)

# , Linear, Module

from eva.models import module, _utils
from eva.metrics import core as metrics_lib
from eva.models.typings import ModelType


INPUT_BATCH = TypeVar("INPUT_BATCH")
"""The input batch type."""


class FeatureExtractor(module.ModelModule[INPUT_BATCH]):
    """@"""

    def __init__(
        self,
        head: ModelType,
        backbone: ModelType | None = None,
        criterion: Callable[..., torch.Tensor] = nn.CrossEntropyLoss,
        optimizer: OptimizerCallable | None = None,
        lr_scheduler: LRSchedulerCallable | None = None,
        metrics: metrics_lib.MetricsSchema | None = None,
    ) -> None:
        """Initializes the linear classification module.

        Args:
            metrics: The metrics schema. Defaults to `None`.
        """
        super().__init__(metrics=metrics)

        self.backbone = backbone
        self.head = head  # nn.Linear(feature_dim, num_classes)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion

    @override
    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.optimizer(self.parameters())
        lr_scheduler = self.lr_scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    @override
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.backbone is not None:
            tensor = self.model.forward(tensor)
        features = tensor.flatten(start_dim=1)
        return self.head(features)

    @override
    def on_fit_start(self) -> None:
        if self.backbone is not None:
            _utils.deactivate_requires_grad(model=self.backbone)

    @override
    def on_fit_end(self) -> None:
        if self.backbone is not None:
            _utils.activate_requires_grad(model=self.backbone)

