""""Neural Network Semantic Segmentation Module."""

from typing import Any, Callable, Tuple

import torch
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim
from torch.optim import lr_scheduler
from typing_extensions import override

from eva.core.metrics import structs as metrics_lib
from eva.core.models.modules import module
from eva.core.models.modules.typings import INPUT_BATCH, INPUT_TENSOR_BATCH
from eva.core.models.modules.utils import batch_postprocess, grad
from eva.vision.models.networks import decoders, encoders


class SemanticSegmentationModule(module.ModelModule):
    """Neural network semantic segmentation module for training on patch embeddings."""

    def __init__(
        self,
        decoder: decoders.Decoder,
        criterion: Callable[..., torch.Tensor],
        encoder: encoders.Encoder | None = None,
        freeze_encoder: bool = True,
        optimizer: OptimizerCallable = optim.Adam,
        lr_scheduler: LRSchedulerCallable = lr_scheduler.ConstantLR,
        metrics: metrics_lib.MetricsSchema | None = None,
        postprocess: batch_postprocess.BatchPostProcess | None = None,
    ) -> None:
        """Initializes the neural net head module.

        Args:
            decoder: The decoder model.
            criterion: The loss function to use.
            encoder: The encoder model. If `None`, it will be expected
                that the input batch returns the features directly.
            freeze_encoder: Whether to freeze the encoder.
            optimizer: The optimizer to use.
            lr_scheduler: The learning rate scheduler to use.
            metrics: The metric groups to track.
            postprocess: A list of helper functions to apply after the
                loss and before the metrics calculation to the model
                predictions and targets.
        """
        super().__init__(metrics=metrics, postprocess=postprocess)

        self.decoder = decoder
        self.criterion = criterion
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    @override
    def configure_optimizers(self) -> Any:
        parameters = self.parameters()
        optimizer = self.optimizer(parameters)
        lr_scheduler = self.lr_scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    @override
    def forward(
        self,
        tensor: torch.Tensor,
        image_size: Tuple[int, int] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if self.encoder is None:
            return self.decoder(tensor, image_size)

        patch_embeddings = self.encoder(tensor)
        return self.decoder(patch_embeddings, tensor.shape[-2:])

    @override
    def on_fit_start(self) -> None:
        if self.encoder is not None and self.freeze_encoder:
            grad.deactivate_requires_grad(self.encoder)

    @override
    def training_step(self, batch: INPUT_TENSOR_BATCH, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self._batch_step(batch)

    @override
    def validation_step(self, batch: INPUT_TENSOR_BATCH, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self._batch_step(batch)

    @override
    def test_step(self, batch: INPUT_TENSOR_BATCH, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self._batch_step(batch)

    @override
    def predict_step(self, batch: INPUT_BATCH, *args: Any, **kwargs: Any) -> torch.Tensor:
        tensor = INPUT_BATCH(*batch).data
        return tensor if self.backbone is None else self.backbone(tensor)

    @override
    def on_fit_end(self) -> None:
        if self.encoder is not None and self.freeze_encoder:
            grad.activate_requires_grad(self.encoder)

    def _batch_step(self, batch: INPUT_TENSOR_BATCH) -> STEP_OUTPUT:
        """Performs a model forward step and calculates the loss.

        Args:
            batch: The desired batch to process.

        Returns:
            The batch step output.
        """
        data, targets, metadata = INPUT_TENSOR_BATCH(*batch)
        predictions = self(data, targets.shape[-2:])
        loss = self.criterion(predictions, targets)
        return {
            "loss": loss,
            "targets": targets,
            "predictions": predictions,
            "metadata": metadata,
        }
