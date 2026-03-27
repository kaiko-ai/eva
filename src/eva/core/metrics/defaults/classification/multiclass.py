"""Default metric collection for multiclass classification tasks."""

from typing import Literal

import torch
from torchmetrics import classification

from eva.core.metrics import structs


class MulticlassClassificationMetrics(structs.MetricCollection):
    """Default metrics for multi-class classification tasks."""

    def __init__(
        self,
        num_classes: int,
        average: Literal["micro", "macro", "weighted", "none"] = "macro",
        ignore_index: int | None = None,
        prefix: str | None = None,
        postfix: str | None = None,
        input_type: Literal["logits", "discrete"] = "logits",
    ) -> None:
        """Initializes the multi-class classification metrics.

        Args:
            num_classes: Integer specifying the number of classes.
            average: Defines the reduction that is applied over labels.
            ignore_index: Specifies a target value that is ignored and does not
                contribute to the metric calculation.
            prefix: A string to append in front of the keys of the output dict.
            postfix: A string to append after the keys of the output dict.
            input_type: Type of input predictions - "logits" for probabilities/logits
                or "discrete" for discrete class predictions. Determines which metrics
                are applicable.
        """
        metrics = [
            classification.MulticlassAccuracy(
                num_classes=num_classes,
                average=average,
                ignore_index=ignore_index,
            ),
            classification.MulticlassF1Score(
                num_classes=num_classes,
                average=average,
                ignore_index=ignore_index,
            ),
            classification.MulticlassPrecision(
                num_classes=num_classes,
                average=average,
                ignore_index=ignore_index,
            ),
            classification.MulticlassRecall(
                num_classes=num_classes,
                average=average,
                ignore_index=ignore_index,
            ),
        ]

        compute_groups = [
            [
                "MulticlassAccuracy",
                "MulticlassF1Score",
                "MulticlassPrecision",
                "MulticlassRecall",
            ]
        ]

        if input_type == "logits" and average != "micro":
            metrics.append(
                classification.MulticlassAUROC(
                    num_classes=num_classes,
                    average=average,
                    ignore_index=ignore_index,
                )
            )
            compute_groups.append(["MulticlassAUROC"])

        super().__init__(
            metrics=metrics,
            prefix=prefix,
            postfix=postfix,
            compute_groups=compute_groups,
        )
        # Set after torchmetrics init to avoid potential state reset side-effects.
        self._ignore_index = ignore_index

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Updates metrics, filtering out predictions matching ignore_index.

        torchmetrics' ignore_index only filters targets, not predictions.
        This causes torch.bincount to crash on negative prediction values
        (e.g. -1 from missing_answer). We filter both here.

        Args:
            preds: Model predictions.
            target: Ground truth labels.
        """
        if self._ignore_index is not None:
            valid_mask = preds != self._ignore_index
            if not valid_mask.any():
                return
            preds, target = preds[valid_mask], target[valid_mask]
        super().update(preds, target)
