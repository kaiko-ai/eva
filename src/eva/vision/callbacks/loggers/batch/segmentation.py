"""Segmentation datasets related data loggers."""

from typing import List, Tuple

import torch
import torchvision
from lightning import pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.nn import functional
from typing_extensions import override

from eva.core.loggers import log
from eva.core.models.modules.typings import INPUT_TENSOR_BATCH
from eva.core.utils import to_cpu
from eva.vision.callbacks.loggers.batch import base
from eva.vision.utils import colormap, convert


class SemanticSegmentationLogger(base.BatchLogger):
    """Log the segmentation batch."""

    def __init__(
        self,
        max_samples: int = 10,
        number_of_images_per_subgrid_row: int = 1,
        log_images: bool = True,
        mean: Tuple[float, ...] = (0.0, 0.0, 0.0),
        std: Tuple[float, ...] = (1.0, 1.0, 1.0),
        log_every_n_epochs: int | None = None,
        log_every_n_steps: int | None = None,
    ) -> None:
        """Initializes the callback object.

        Args:
            max_samples: The maximum number of images displayed in the grid.
            number_of_images_per_subgrid_row: Number of images displayed in each
                row of each sub-grid (that is images, targets and predictions).
            log_images: Whether to log the input batch images.
            mean: The mean of the input images to de-normalize from.
            std: The std of the input images to de-normalize from.
            log_every_n_epochs: Epoch-wise logging frequency.
            log_every_n_steps: Step-wise logging frequency.
        """
        super().__init__(
            log_every_n_epochs=log_every_n_epochs,
            log_every_n_steps=log_every_n_steps,
        )

        self._max_samples = max_samples
        self._number_of_images_per_subgrid_row = number_of_images_per_subgrid_row
        self._log_images = log_images
        self._mean = mean
        self._std = std

    @override
    def _log_batch(
        self,
        trainer: pl.Trainer,
        outputs: STEP_OUTPUT,
        batch: INPUT_TENSOR_BATCH,
        tag: str,
    ) -> None:
        predictions = outputs.get("predictions") if isinstance(outputs, dict) else None
        if predictions is None:
            raise ValueError("Key `predictions` is missing from the output.")

        data_batch, target_batch = batch[0], batch[1]
        data, targets, predictions = _subsample_tensors(
            tensors_stack=[data_batch, target_batch, predictions],
            max_samples=self._max_samples,
        )
        data, targets, predictions = to_cpu([data, targets, predictions])
        predictions = torch.argmax(predictions, dim=1)

        target_images = list(map(_draw_semantic_mask, targets))
        prediction_images = list(map(_draw_semantic_mask, predictions))
        image_groups = [target_images, prediction_images]

        if self._log_images:
            images = list(map(self._format_image, data))
            overlay_targets = [
                _overlay_mask(image, mask) for image, mask in zip(images, targets, strict=False)
            ]
            overlay_predictions = [
                _overlay_mask(image, mask) for image, mask in zip(images, predictions, strict=False)
            ]
            image_groups = [images, overlay_targets, overlay_predictions] + image_groups

        image_grid = _make_grid_from_image_groups(
            image_groups, self._number_of_images_per_subgrid_row
        )

        log.log_image(
            trainer.loggers,
            image=image_grid,
            tag=tag,
            step=trainer.global_step,
        )

    def _format_image(self, image: torch.Tensor) -> torch.Tensor:
        """Descaled an image tensor to (0, 255) uint8 tensor."""
        return convert.descale_and_denorm_image(image, mean=self._mean, std=self._std)


def _subsample_tensors(
    tensors_stack: List[torch.Tensor],
    max_samples: int,
) -> List[torch.Tensor]:
    """Sub-samples tensors from a list of tensors in-place.

    Args:
        tensors_stack: A list of tensors.
        max_samples: The maximum number of images
            displayed in the grid.

    Returns:
        A sub-sample of the input tensors stack.
    """
    for i, tensor in enumerate(tensors_stack):
        tensors_stack[i] = tensor[:max_samples]
    return tensors_stack


def _draw_semantic_mask(tensor: torch.Tensor) -> torch.Tensor:
    """Draws a semantic mask to an image RGB tensor.

    The input semantic mask is a (H x W) shaped tensor with
    integer values which represent the pixel class id.

    Args:
        tensor: An image tensor of range [0., N_CLASSES].

    Returns:
        The image as a tensor of range [0., 255.].
    """
    tensor = torch.squeeze(tensor)
    height, width = tensor.shape[-2], tensor.shape[-1]
    red, green, blue = torch.zeros((3, height, width), dtype=torch.uint8)
    class_ids = torch.unique(tensor)
    colors = colormap.get_colors(max(class_ids))
    for class_id in class_ids:
        indices = tensor == class_id
        red[indices], green[indices], blue[indices] = colors[int(class_id)]
    return torch.stack([red, green, blue])


def _overlay_mask(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Overlays a segmentation mask onto an image.

    Args:
        image: A 3D tensor of shape (C, H, W) representing the image.
        mask: A 2D tensor of shape (H, W) representing the segmentation mask.
            Each pixel in the mask corresponds to a class label.

    Returns:
        A tensor of the same shape as the input image (C, H, W) with the
        segmentation mask overlaid on top. The output image retains the
        original color channels but with the mask applied, using the colors
        from the predefined colormap.
    """
    binary_masks = functional.one_hot(mask).permute(2, 0, 1).to(dtype=torch.bool)
    colors = colormap.get_colors(binary_masks.shape[0] + 1)
    return torchvision.utils.draw_segmentation_masks(
        image, binary_masks[1:], alpha=0.65, colors=colors[1:]  # type: ignore
    )


def _make_grid_from_image_groups(
    image_groups: List[List[torch.Tensor]],
    number_of_images_per_subgrid_row: int = 2,
) -> torch.Tensor:
    """Creates a single image grid from image groups.

    For example, it can combine the input images, targets predictions into a single image.

    Args:
        image_groups: A list of lists of image tensors of shape (C x H x W)
            all of the same size.
        number_of_images_per_subgrid_row: Number of images displayed in each
            row of the sub-grid.

    Returns:
        An image grid as a `torch.Tensor`.
    """
    return torchvision.utils.make_grid(
        [
            torchvision.utils.make_grid(image_group, nrow=number_of_images_per_subgrid_row)
            for image_group in image_groups
        ],
        nrow=len(image_groups),
    )
