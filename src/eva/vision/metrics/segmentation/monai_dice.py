"""Wrapper for dice score metric from MONAI."""

from typing import Literal

from monai.metrics.meandice import DiceMetric
from typing_extensions import override

from eva.vision.metrics import wrappers
from eva.vision.metrics.segmentation import _utils


class MonaiDiceScore(wrappers.MonaiMetricWrapper):
    """Wrapper to make MONAI's `DiceMetric` compatible with `torchmetrics`."""

    def __init__(
        self,
        num_classes: int,
        include_background: bool = True,
        input_format: Literal["one-hot", "index"] = "index",
        reduction: str = "mean",
        ignore_index: int | None = None,
        **kwargs,
    ):
        """Initializes metric.

        Args:
            num_classes: The number of classes in the dataset.
            include_background: Whether to include the background class in the computation.
            reduction: The method to reduce the dice score. Options are `"mean"`, `"sum"`, `"none"`.
            input_format: Choose between "one-hot" for one-hot encoded tensors or "index"
                for index tensors.
            ignore_index: Integer specifying a target class to ignore. If given, this class
                index does not contribute to the returned score.
            kwargs: Additional keyword arguments for instantiating monai's `DiceMetric` class.
        """
        super().__init__(
            DiceMetric(
                include_background=include_background or (ignore_index == 0),
                reduction=reduction,
                num_classes=num_classes - (ignore_index is not None),
                **kwargs,
            )
        )

        self.reduction = reduction
        self.orig_num_classes = num_classes
        self.ignore_index = ignore_index
        self.input_format = input_format

    @override
    def update(self, preds, target):
        if self.input_format == "index":
            preds = _utils.index_to_one_hot(preds, num_classes=self.orig_num_classes)
            target = _utils.index_to_one_hot(target, num_classes=self.orig_num_classes)
        if self.ignore_index is not None:
            preds, target = _utils.apply_ignore_index(preds, target, self.ignore_index)
        return super().update(preds, target)

    @override
    def compute(self):
        result = super().compute()
        if self.reduction == "none" and len(result) > 1:
            result = result.nanmean(dim=0)
        return result
