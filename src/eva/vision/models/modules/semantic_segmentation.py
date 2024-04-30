""""Neural Network Semantic Segmentation Module."""

from typing import Any, Callable, Tuple, Iterable

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
        lr_multiplier_encoder: float = 0.0,
        optimizer: OptimizerCallable = optim.AdamW,
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
            lr_multiplier_encoder: The learning rate multiplier for the
                encoder parameters. If `0`, it will freeze the encoder.
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
        self.lr_multiplier_encoder = lr_multiplier_encoder
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    @override
    def configure_model(self) -> None:
        if self.encoder is not None and self.lr_multiplier_encoder == 0:
            self._freeze_encoder()

    @override
    def configure_optimizers(self) -> Any:
        optimizer = self.optimizer([
            {"params": self._decoder_params},
            {
                "params": self._encoder_params,
                "lr": self._base_lr * self.lr_multiplier_encoder,
            },
        ])
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
        """Maps the input tensor (image tensor or embeddings) to masks.

        If `tensor` is image tensor, then the `self.encoder`
        should be implemented, otherwise it will be interpreted
        as embeddings, where the `image_size` should be given.
        """
        if self.encoder is None and image_size is None:
            raise ValueError(
                "Please provide the expected `image_size` that the "
                "decoder should map the embeddings (`tensor`) to."
            )

        patch_embeddings = self.encoder(tensor) if self.encoder else tensor
        return self.decoder(patch_embeddings, image_size or tensor.shape[-2:])

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

    def _freeze_encoder(self) -> None:
        """Freezes the encoder network."""
        grad.deactivate_requires_grad(self.encoder)

    @property
    def _base_lr(self) -> float:
        """Returns the base learning rate."""
        base_optimizer = self.optimizer(self.parameters())
        return base_optimizer.param_groups[-1]["lr"]

    @property
    def _encoder_params(self) -> Iterable[torch.Tensor]:
        """Returns the trainable parameters of the encoder."""
        return filter(lambda p: p.requires_grad, self.encoder.parameters())

    @property
    def _decoder_params(self) -> Iterable[torch.Tensor]:
        """Returns the trainable parameters of the decoder."""
        return filter(lambda p: p.requires_grad, self.decoder.parameters())
