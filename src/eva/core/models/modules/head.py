"""Neural Network Head Module."""

from typing import Any, Callable, Dict, List

import torch
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn, optim
from torch.optim import lr_scheduler
from typing_extensions import override

from eva.core.metrics import structs as metrics_lib
from eva.core.models.modules import module
from eva.core.models.modules.typings import INPUT_BATCH, MODEL_TYPE
from eva.core.models.modules.utils import batch_postprocess, grad, submodule_state_dict
from eva.core.utils import parser


class HeadModule(module.ModelModule):
    """Neural Net Head Module for training on features.

    It can be used for supervised (mini-batch) stochastic gradient descent
    downstream tasks such as classification, regression and segmentation.
    """

    def __init__(
        self,
        head: Dict[str, Any] | MODEL_TYPE,
        criterion: Callable[..., torch.Tensor],
        backbone: MODEL_TYPE | None = None,
        optimizer: OptimizerCallable = optim.Adam,
        lr_scheduler: LRSchedulerCallable = lr_scheduler.ConstantLR,
        metrics: metrics_lib.MetricsSchema | None = None,
        postprocess: batch_postprocess.BatchPostProcess | None = None,
        save_head_only: bool = True,
    ) -> None:
        """Initializes the neural net head module.

        Args:
            head: The neural network that would be trained on the features.
                If its a dictionary, it will be parsed to an object during the
                `configure_model` step.
            criterion: The loss function to use.
            backbone: The feature extractor. If `None`, it will be expected
                that the input batch returns the features directly.
            optimizer: The optimizer to use.
            lr_scheduler: The learning rate scheduler to use.
            metrics: The metric groups to track.
            postprocess: A list of helper functions to apply after the
                loss and before the metrics calculation to the model
                predictions and targets.
            save_head_only: Whether to save only the head during checkpointing. If False,
                will also save the backbone (not recommended when frozen).
        """
        super().__init__(metrics=metrics, postprocess=postprocess)

        self.head = head  # type: ignore
        self.criterion = criterion
        self.backbone = backbone
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.save_head_only = save_head_only

    @override
    def configure_model(self) -> Any:
        if self.backbone is not None:
            grad.deactivate_requires_grad(self.backbone)

        if isinstance(self.head, dict):
            self.head: MODEL_TYPE = parser.parse_object(self.head, expected_type=nn.Module)

    @override
    def configure_optimizers(self) -> Any:
        parameters = self.head.parameters()
        optimizer = self.optimizer(parameters)
        lr_scheduler = self.lr_scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    @override
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.save_head_only:
            checkpoint["state_dict"] = submodule_state_dict(checkpoint["state_dict"], "head")
        super().on_save_checkpoint(checkpoint)

    @override
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.save_head_only and self.backbone is not None:
            checkpoint["state_dict"].update(
                {f"backbone.{k}": v for k, v in self.backbone.state_dict().items()}
            )
        super().on_load_checkpoint(checkpoint)

    @override
    def forward(self, tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        features = tensor if self.backbone is None else self.backbone(tensor)
        return self.head(features).squeeze(-1)

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
    def predict_step(
        self, batch: INPUT_BATCH, *args: Any, **kwargs: Any
    ) -> torch.Tensor | List[torch.Tensor]:
        tensor = INPUT_BATCH(*batch).data
        return tensor if self.backbone is None else self.backbone(tensor)

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
