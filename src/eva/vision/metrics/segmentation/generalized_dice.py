"""Generalized Dice Score metric for semantic segmentation."""

from typing import Any, Literal

import torch
from torchmetrics import segmentation
from typing_extensions import override

from eva.vision.metrics.segmentation import _utils


class GeneralizedDiceScore(segmentation.GeneralizedDiceScore):
    """Defines the Generalized Dice Score.

    It expands the `torchmetrics` class by including an `ignore_index`
    functionality and converting tensors to one-hot format.
    """

    def __init__(
        self,
        num_classes: int,
        include_background: bool = True,
        weight_type: Literal["square", "simple", "linear"] = "linear",
        ignore_index: int | None = None,
        per_class: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initializes the metric.

        Args:
            num_classes: The number of classes in the segmentation problem.
            include_background: Whether to include the background class in the computation
            weight_type: The type of weight to apply to each class. Can be one of `"square"`,
                `"simple"`, or `"linear"`.
            ignore_index: Integer specifying a target class to ignore. If given, this class
                index does not contribute to the returned score, regardless of reduction method.
            per_class: Whether to compute the IoU for each class separately. If set to ``False``,
                the metric will compute the mean IoU over all classes.
            kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.
        """
        super().__init__(
            num_classes=num_classes
            - (ignore_index is not None)
            + (ignore_index == 0 and not include_background),
            include_background=include_background,
            weight_type=weight_type,
            per_class=per_class,
            **kwargs,
        )
        self.orig_num_classes = num_classes
        self.ignore_index = ignore_index

    @override
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds = _utils.index_to_one_hot(preds, num_classes=self.orig_num_classes)
        target = _utils.index_to_one_hot(target, num_classes=self.orig_num_classes)
        if self.ignore_index is not None:
            preds, target = _utils.apply_ignore_index(preds, target, self.ignore_index)
        super().update(preds=preds.long(), target=target.long())
