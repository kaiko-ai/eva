"""Monai metrics wrappers."""

import torch
import torchmetrics
from monai.metrics.metric import CumulativeIterationMetric
from typing_extensions import override


class MonaiMetricWrapper(torchmetrics.Metric):
    """Wrapper class to make MONAI metrics compatible with `torchmetrics`."""

    def __init__(self, monai_metric: CumulativeIterationMetric):
        """Initializes the monai metric wrapper.

        Args:
            monai_metric: The MONAI metric to wrap.
        """
        super().__init__()
        self._monai_metric = monai_metric

    @override
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self._monai_metric(preds, target)

    @override
    def compute(self) -> torch.Tensor:
        return self._monai_metric.aggregate()

    @override
    def reset(self) -> None:
        super().reset()
        self._monai_metric.reset()
