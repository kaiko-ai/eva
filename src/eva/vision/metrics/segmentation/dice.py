"""Dice Score metric for semantic segmentation."""

from typing import Any, Literal

import torch
from torchmetrics import segmentation
from typing_extensions import override

from eva.vision.metrics.segmentation import _utils


class DiceScore(segmentation.DiceScore):
    """Dice Score metric for semantic segmentation tasks.

    This implementation expands the `torchmetrics` class by including
    an `ignore_index` functionality and converting tensors to one-hot
    format on the fly.
    """

    def __init__(
        self,
        num_classes: int,
        include_background: bool = True,
        average: Literal["micro", "macro", "weighted", "none"] | None = "micro",
        input_format: Literal["one-hot", "index", "auto"] = "auto",
        ignore_index: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the metric.

        Args:
            num_classes: The number of classes in the segmentation problem.
            include_background: Whether to include the background class in the computation
            average: The method to average the dice score accross the different classes. Options are
                `"micro"`, `"macro"`, `"weighted"`, `"none"` or `None`.
            input_format: Input tensor format. Options are `"one-hot"` for one-hot encoded tensors,
                `"index"` for index tensors, or `"auto"` to automatically convert the format
                to one-hot.
            ignore_index: Integer specifying a target class to ignore. If given, this class
                index does not contribute to the returned score, regardless of reduction method.
            kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.
        """
        super().__init__(
            num_classes=num_classes
            - (ignore_index is not None)
            + (ignore_index == 0 and not include_background),
            include_background=include_background,
            average=average,
            input_format=input_format if input_format == "index" else "one-hot",
            **kwargs,
        )
        self.orig_num_classes = num_classes
        self.ignore_index = ignore_index
        self.input_format = input_format

    @override
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if self.input_format == "auto":
            preds = _utils.index_to_one_hot(preds, num_classes=self.orig_num_classes)
            target = _utils.index_to_one_hot(target, num_classes=self.orig_num_classes)
        if self.ignore_index is not None:
            if self.input_format == "index":
                raise ValueError(
                    "When `ignore_index` is set, `input_format` must be 'one-hot' or 'auto'."
                )
            preds, target = _utils.apply_ignore_index(preds, target, self.ignore_index)

        super().update(preds.long(), target.long())
