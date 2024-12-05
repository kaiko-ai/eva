""""Neural Network Semantic Segmentation Module."""

from typing import Any, Callable, Dict, Iterable, List, Tuple

import torch
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn, optim
from torch.optim import lr_scheduler
from typing_extensions import override

from eva.core.metrics import structs as metrics_lib
from eva.core.models.modules import module
from eva.core.models.modules.typings import INPUT_BATCH, INPUT_TENSOR_BATCH
from eva.core.models.modules.utils import batch_postprocess, grad
from eva.core.utils import parser
from eva.vision.models.networks import decoders
from eva.vision.models.networks.decoders.segmentation.typings import DecoderInputs


class SemanticSegmentationModule(module.ModelModule):
    """Neural network semantic segmentation module for training on patch embeddings."""

    def __init__(
        self,
        decoder: decoders.Decoder,
        criterion: Callable[..., torch.Tensor],
        encoder: Dict[str, Any] | Callable[[torch.Tensor], List[torch.Tensor]] | None = None,
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
                If pass as a dictionary, it will be parsed to an object
                during the `configure_model` step.
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
        self.encoder = encoder  # type: ignore
        self.lr_multiplier_encoder = lr_multiplier_encoder
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    @override
    def configure_model(self) -> None:
        self._freeze_encoder()

        if isinstance(self.encoder, dict):
            self.encoder: Callable[[torch.Tensor], List[torch.Tensor]] = parser.parse_object(
                self.encoder,
                expected_type=nn.Module,
            )

    @override
    def configure_optimizers(self) -> Any:
        optimizer = self.optimizer(
            [
                {"params": self.decoder.parameters()},
                {
                    "params": self._encoder_trainable_parameters(),
                    "lr": self._base_lr * self.lr_multiplier_encoder,
                },
            ]
        )
        lr_scheduler = self.lr_scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    @override
    def forward(
        self,
        inputs: torch.Tensor,
        to_size: Tuple[int, int] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Maps the input tensor (image tensor or embeddings) to masks.

        If `inputs` is image tensor, then the `self.encoder`
        should be implemented, otherwise it will be interpreted
        as embeddings, where the `to_size` should be given.
        """
        if self.encoder is None and to_size is None:
            raise ValueError(
                "Please provide the expected `to_size` that the "
                "decoder should map the embeddings (`inputs`) to."
            )
        features = self.encoder(inputs) if self.encoder else inputs
        decoder_inputs = DecoderInputs(features, to_size or inputs.shape[-2:], inputs)  # type: ignore
        return self.decoder(decoder_inputs)

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
        return self.encoder(tensor) if isinstance(self.encoder, nn.Module) else tensor

    @property
    def _base_lr(self) -> float:
        """Returns the base learning rate."""
        base_optimizer = self.optimizer(self.parameters())
        return base_optimizer.param_groups[-1]["lr"]

    def _encoder_trainable_parameters(self) -> Iterable[torch.Tensor]:
        """Returns the trainable parameters of the encoder."""
        return (
            self.encoder.parameters()
            if isinstance(self.encoder, nn.Module) and self.lr_multiplier_encoder > 0
            else iter(())
        )

    def _freeze_encoder(self) -> None:
        """If initialized, it freezes the encoder network."""
        if isinstance(self.encoder, nn.Module) and self.lr_multiplier_encoder == 0:
            grad.deactivate_requires_grad(self.encoder)

    def _batch_step(self, batch: INPUT_TENSOR_BATCH) -> STEP_OUTPUT:
        """Performs a model forward step and calculates the loss.

        Args:
            batch: The desired batch to process.

        Returns:
            The batch step output.
        """
        data, targets, metadata = INPUT_TENSOR_BATCH(*batch)
        predictions = self(data, to_size=targets.shape[-2:])
        loss = self.criterion(predictions, targets)
        return {
            "loss": loss,
            "targets": targets,
            "predictions": predictions,
            "metadata": metadata,
        }
