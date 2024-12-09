"""Generalized Dice Score metric for semantic segmentation."""

from typing import Any, Literal

import torch
from torchmetrics import segmentation
from torchmetrics.functional.segmentation.dice import _dice_score_update
from typing_extensions import override

from eva.vision.metrics.segmentation import _utils


class DiceScore(segmentation.DiceScore):
    """Defines the Generalized Dice Score.

    It expands the `torchmetrics` class by including an `ignore_index`
    functionality and converting tensors to one-hot format.
    """

    def __init__(
        self,
        num_classes: int,
        include_background: bool = True,
        average: Literal["micro", "macro", "weighted", "none"] | None = "micro",
        ignore_index: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the metric.

        Args:
            num_classes: The number of classes in the segmentation problem.
            include_background: Whether to include the background class in the computation
            average: The method to average the dice score accross the different classes. Options are
                `"micro"`, `"macro"`, `"weighted"`, `"none"` or `None`.
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
            input_format="one-hot",
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

        # TODO: Replace _update by super.update() once the following issue is fixed:
        # https://github.com/Lightning-AI/torchmetrics/issues/2847
        self._update(preds.long(), target.long())
        # super().update(preds=preds.long(), target=target.long())

    def _update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        numerator, denominator, support = _dice_score_update(
            preds, target, self.num_classes, self.include_background, self.input_format  # type: ignore
        )
        self.numerator.append(numerator)
        self.denominator.append(denominator)
        self.support.append(support)
