"""Default metric collection for multiclass semantic segmentation tasks."""

from typing import Literal

from torchmetrics import classification

from eva.core.metrics import structs


class MulticlassSegmentationMetrics(structs.MetricCollection):
    """Default metrics for multi-class semantic segmentation tasks."""

    def __init__(
        self,
        num_classes: int,
        average: Literal["macro", "weighted", "none"] = "macro",
        ignore_index: int | None = None,
        prefix: str | None = None,
        postfix: str | None = None,
    ) -> None:
        """Initializes the multi-class semantic segmentation metrics.

        Args:
            num_classes: Integer specifying the number of classes.
            average: Defines the reduction that is applied over labels.
            ignore_index: Specifies a target value that is ignored and
                does not contribute to the metric calculation.
            prefix: A string to add before the keys in the output dictionary.
            postfix: A string to add after the keys in the output dictionary.
        """
        super().__init__(
            metrics=[
                classification.MulticlassJaccardIndex(
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
                    "MulticlassJaccardIndex",
                ],
                [
                    "MulticlassF1Score",
                    "MulticlassPrecision",
                    "MulticlassRecall",
                ],
            ],
        )
