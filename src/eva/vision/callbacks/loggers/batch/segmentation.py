"""Segmentation datasets related data loggers."""

from typing import Iterable, List, Tuple

import torch
import torchvision
from lightning import pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchvision.transforms.v2 import functional
from typing_extensions import override

from eva.core.models.modules.typings import INPUT_TENSOR_BATCH
from eva.core.utils import to_cpu
from eva.vision.callbacks.loggers.batch import base
from eva.vision.utils import colormap


class SemanticSegmentationLogger(base.BatchLogger):
    """Log the segmentation batch."""

    def __init__(
        self,
        max_samples: int = 10,
        number_of_rows: int | None = None,
        mean: Tuple[float, ...] = (0.5, 0.5, 0.5),
        std: Tuple[float, ...] = (0.5, 0.5, 0.5),
        log_every_n_epochs: int | None = 1,
        log_every_n_steps: int | None = None,
    ) -> None:
        """Initializes the callback object.

        Args:
            max_samples: The maximum number of images displayed in the grid.
            number_of_rows: Number of images displayed in each row of the grid.
                If `None`, it is the sqrt(total_images).
            mean: The mean of the input images.
            std: The std of the input images.
            log_every_n_epochs: Epoch-wise logging frequency.
            log_every_n_steps: Step-wise logging frequency.
        """
        super().__init__(
            log_every_n_epochs=log_every_n_epochs,
            log_every_n_steps=log_every_n_steps,
        )

        self._max_samples = max_samples
        self._number_of_rows = number_of_rows
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

        images, targets, predictions = _subsample_tensors(
            tensors_stack=[batch[0], batch[1], predictions],
            max_samples=self._max_samples,
        )
        images, targets, predictions = to_cpu([images, targets, predictions])

        images = [_denormalize_image(image, mean=self._mean, std=self._std) for image in images]
        targets = list(map(_draw_semantic_mask, targets))
        predictions = list(map(_draw_semantic_mask, torch.argmax(predictions, dim=1)))
        image_grid = _make_grid_from_outputs(images, targets, predictions)

        from torchvision.utils import save_image

        save_image(image_grid.to(dtype=torch.float32), f"{tag}_{trainer.global_step}.png")


def _subsample_tensors(tensors_stack: List[torch.Tensor], max_samples: int) -> List[torch.Tensor]:
    """Sub-samples tensors from a list of tensors in-place.

    Args:
        tensors_stack: A list of tensors.
        max_samples: The maximum number of images displayed in the grid.
            Defaults to `16`.

    Returns:
        A sub-sample of the input tensors stack.
    """
    for i, tensor in enumerate(tensors_stack):
        tensors_stack[i] = tensor[:max_samples]
    return tensors_stack


def _denormalize_image(
    tensor: torch.Tensor,
    mean: Iterable[float] = (0.5, 0.5, 0.5),
    std: Iterable[float] = (0.5, 0.5, 0.5),
    inplace: bool = True,
) -> torch.Tensor:
    """De-normalizes an image tensor to (0., 1.) range.

    Args:
        tensor: An image float tensor.
        mean: The normalized channels mean values.
        std: The normalized channels std values.
        inplace: Whether to perform the operation in-place.
            Defaults to `True`.

    Returns:
        The de-normalized image tensor of range (0., 1.).
    """
    if not inplace:
        tensor = tensor.clone()

    return functional.normalize(
        tensor,
        mean=[-cmean / cstd for cmean, cstd in zip(mean, std, strict=False)],
        std=[1 / cstd for cstd in std],
    )


def _draw_semantic_mask(tensor: torch.Tensor) -> torch.Tensor:
    """Draws a semantic mask to an image RGB tensor.

    The input semantic mask is a (H x W) shaped tensor with integer values
    which represent the pixel class id.

    Args:
        tensor: An image tensor of range [0., 1.].

    Returns:
        The image as a numpy tensor in the range of [0., 255.].
    """
    if tensor.dtype == torch.uint8:
        return tensor

    tensor = torch.squeeze(tensor)

    height, width = tensor.shape[-2], tensor.shape[-1]
    red, green, blue = torch.zeros((3, height, width), dtype=torch.uint8)
    for class_id, color in colormap.COLORMAP.items():
        indices = tensor == class_id
        red[indices], green[indices], blue[indices] = color

    return torch.stack([red, green, blue])


def _make_grid_from_outputs(
    images: List[torch.Tensor],
    targets: List[torch.Tensor],
    predictions: List[torch.Tensor],
    nrows: int = 2,
) -> torch.Tensor:
    """Creates a single image grid from the batch output.

    It combines the input images, targets predictions into a single image.

    Returns:
        An image grid as a `torch.Tensor`.
    """
    return torchvision.utils.make_grid(
        [
            torchvision.utils.make_grid(images, nrow=nrows),
            torchvision.utils.make_grid(targets, nrow=nrows),
            torchvision.utils.make_grid(predictions, nrow=nrows),
        ],
        nrow=3,
    )
