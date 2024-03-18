"""Implementation of the average loss metric."""

import torch
from loguru import logger
from typing_extensions import override

from eva.core.metrics import structs


class AverageLoss(structs.Metric):
    """Average loss metric tracker."""

    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(self) -> None:
        """Initializes the metric."""
        super().__init__()

        self.add_state("value", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    @override
    def update(self, loss: torch.Tensor) -> None:
        _check_nans(loss)
        total_samples = loss.numel()
        if total_samples == 0:
            return

        self.value = self.value + torch.sum(loss)
        self.total = self.total + total_samples

    @override
    def compute(self) -> torch.Tensor:
        return self.value / self.total


def _check_nans(tensor: torch.Tensor) -> None:
    """Checks for nan values and raises a warning.

    Raises:
        Warning: If the input tensor consists of any NaN(s).
    """
    nan_values = tensor.isnan()
    if nan_values.any():
        logger.warning("Encountered `nan` value(s) in input tensor.")
