"""Default metric collection for binary classification tasks."""

from torchmetrics import classification

from eva import metrics
from eva.metrics import core


class BinaryClassificationMetrics(core.MetricCollection):
    """Default metrics for multi-class classification tasks."""

    def __init__(
        self,
        threshold: float = 0.5,
        ignore_index: int | None = None,
        prefix: str | None = None,
        postfix: str | None = None,
    ) -> None:
        """Initializes the binary classification metrics.

        Args:
            threshold: Threshold for transforming probability to binary (0,1) predictions
            ignore_index: Specifies a target value that is ignored and does not
                contribute to the metric calculation.
            prefix: A string to append in front of the keys of the output dict.
            postfix: A string to append after the keys of the output dict.
        """
        super().__init__(
            metrics=[
                classification.BinaryAccuracy(
                    threshold=threshold,
                    ignore_index=ignore_index,
                ),
                metrics.BinaryBalancedAccuracy(
                    threshold=threshold,
                    ignore_index=ignore_index,
                ),
                classification.BinaryPrecision(
                    threshold=threshold,
                    ignore_index=ignore_index,
                ),
                classification.BinaryRecall(
                    threshold=threshold,
                    ignore_index=ignore_index,
                ),
                classification.BinaryF1Score(
                    threshold=threshold,
                    ignore_index=ignore_index,
                ),
                classification.BinaryAUROC(
                    ignore_index=ignore_index,
                ),
            ],
            prefix=prefix,
            postfix=postfix,
            compute_groups=[
                [
                    "BinaryAccuracy",
                    "BinaryPrecision",
                    "BinaryRecall",
                    "BinaryF1Score",
                ],
                [
                    "BinaryBalancedAccuracy",  # needs to be seperate, as derived from multiclass
                ],
                [
                    "BinaryAUROC",
                ],
            ],
        )
