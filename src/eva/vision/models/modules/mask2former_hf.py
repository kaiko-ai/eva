""""Neural Network Semantic Segmentation Module."""

from typing import Any, Callable, Iterable, List, Tuple

import torch
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn, optim
from torch.optim import lr_scheduler
from typing_extensions import override
import transformers

from eva.core.metrics import structs as metrics_lib
from eva.core.models.modules import module
from eva.core.models.modules.typings import INPUT_BATCH, INPUT_TENSOR_BATCH
from eva.core.models.modules.utils import batch_postprocess, grad
from eva.vision import losses
from eva.vision.models.networks.decoders import segmentation


class Mask2FormerHFModule(module.ModelModule):
    """Mask2Former semantic segmentation module."""

    def __init__(
        self,
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


        # self.model = transformers.Mask2FormerForUniversalSegmentation(
        #     config=transformers.Mask2FormerConfig(
        #         use_timm_backbone=True,
        #         backbone="vit_small_patch16_224",
        #         backbone_kwargs={
        #             "out_indices": 3
        #         },
        #         num_labels=8,
        #     ),
        # )

        self.model = transformers.Mask2FormerForUniversalSegmentation(
            config=transformers.Mask2FormerConfig(
                use_timm_backbone=True,
                backbone="vit_small_patch16_224",
                use_pretrained_backbone=True,
                backbone_kwargs={
                    "features_only": True,
                    "out_indices": 3
                },
                num_labels=8,
                num_queries=100,
                decoder_layers=10,
                num_attention_heads=8,
            ),
        )

        self.cls_seg = transformers.Mask2FormerImageProcessor()

        self.encoder = self.model.model.pixel_level_module.encoder

        self.lr_multiplier_encoder = lr_multiplier_encoder
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    @override
    def configure_model(self) -> None:
        self._freeze_encoder()

    @override
    def configure_optimizers(self) -> Any:
        optimizer = self.optimizer(
            [
                {"params": self.model.parameters()},
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
        outputs = self.model(inputs)
        predictions = torch.stack(self.cls_seg.post_process_semantic_segmentation(outputs, to_size))
        return predictions

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
        mask_labels, class_labels = _convert_semantic_label(targets)
        outputs = self.model(data, mask_labels, class_labels)
        
        from monai.networks import one_hot  # type: ignore
        predictions = self.cls_seg.post_process_semantic_segmentation(outputs, data.shape[0] * [(224, 224)])
        predictions = torch.stack(predictions)
        predictions = one_hot(predictions[:, None, ...], num_classes=8)

        return {
            "loss": outputs.loss,
            "targets": targets,
            "predictions": predictions,
            "metadata": metadata,
        }


def _convert_semantic_label(
    semantic_labels: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Converts semantic labels to mask and class labels.

    Args:
        semantic_labels: Tensor of shape `(batch_size, height, width)`
            representing the semantic labels.

    Returns:
        A tuple containing the list of mask labels and list of class labels.
    """
    mask_labels, class_labels = [], []
    for batch in semantic_labels:
        masks, labels = [], []
        for label_id in batch.unique():
            masks.append((batch == label_id).float())
            labels.append(label_id)
        mask_labels.append(torch.stack(masks))
        class_labels.append(torch.stack(labels))
    return mask_labels, class_labels
