"""Default metric collection for multiclass classification tasks."""

from typing import Literal

from torchmetrics import classification

from eva.core.metrics import structs


class MulticlassClassificationMetrics(structs.MetricCollection):
    """Default metrics for multi-class classification tasks."""

    def __init__(
        self,
        num_classes: int,
        average: Literal["macro", "weighted", "none"] = "macro",
        ignore_index: int | None = None,
        prefix: str | None = None,
        postfix: str | None = None,
    ) -> None:
        """Initializes the multi-class classification metrics.

        The metrics instantiated here are:

        - MulticlassAccuracy
        - MulticlassPrecision
        - MulticlassRecall
        - MulticlassF1Score
        - MulticlassAUROC

        Args:
            num_classes: Integer specifying the number of classes.
            average: Defines the reduction that is applied over labels.
            ignore_index: Specifies a target value that is ignored and does not
                contribute to the metric calculation.
            prefix: A string to append in front of the keys of the output dict.
            postfix: A string to append after the keys of the output dict.
        """
        super().__init__(
            metrics=[
                classification.MulticlassAUROC(
                    num_classes=num_classes,
                    average=average,
                    ignore_index=ignore_index,
                ),
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
            ],
            prefix=prefix,
            postfix=postfix,
            compute_groups=[
                [
                    "MulticlassAccuracy",
                    "MulticlassF1Score",
                    "MulticlassPrecision",
                    "MulticlassRecall",
                ],
                [
                    "MulticlassAUROC",
                ],
            ],
        )
