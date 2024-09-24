"""Generalized Dice Score metric for semantic segmentation."""

from typing import Any, Literal

import torch
from torchmetrics import segmentation
from typing_extensions import override


class GeneralizedDiceScore(segmentation.GeneralizedDiceScore):
    """Defines the Generalized Dice Score.

    It expands the `torchmetrics` class by including an `ignore_index`
    functionality.
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
            input_format: What kind of input the function receives. Choose between ``"one-hot"``
                for one-hot encoded tensors or ``"index"`` for index tensors.
            ignore_index: Integer specifying a target class to ignore. If given, this class
                index does not contribute to the returned score, regardless of reduction method.
            per_class: Whether to compute the IoU for each class separately. If set to ``False``,
                the metric will compute the mean IoU over all classes.
            kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.
        """
        super().__init__(
            num_classes=num_classes,
            include_background=include_background,
            weight_type=weight_type,
            per_class=per_class,
            **kwargs,
        )

        self.ignore_index = ignore_index

    @override
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            mask = mask.all(dim=-1, keepdim=True)
            preds = preds * mask
            target = target * mask

        super().update(preds=preds, target=target)
