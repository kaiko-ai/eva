"""Default metric collection for multiclass semantic segmentation tasks."""

from typing import Literal

from torchmetrics import segmentation

from eva.core.metrics import structs


class MulticlassSegmentationMetrics(structs.MetricCollection):
    """Default metrics for multi-class semantic segmentation tasks."""

    def __init__(
        self,
        num_classes: int,
        include_background: bool = False,
        input_format: Literal["one-hot", "index"] = "index",
        prefix: str | None = None,
        postfix: str | None = None,
    ) -> None:
        """Initializes the multi-class semantic segmentation metrics.

        Args:
            num_classes: Integer specifying the number of classes.
            include_background: Whether to include the background class in the metrics computation.
            input_format: What kind of input the metrics should expect.
            prefix: A string to add before the keys in the output dictionary.
            postfix: A string to add after the keys in the output dictionary.
        """
        super().__init__(
            metrics=[
                segmentation.GeneralizedDiceScore(
                    num_classes=num_classes,
                    include_background=include_background,
                    input_format=input_format,
                    weight_type="linear",
                ),
                segmentation.MeanIoU(
                    num_classes=num_classes,
                    include_background=include_background,
                    input_format=input_format,
                ),
            ],
            prefix=prefix,
            postfix=postfix,
            compute_groups=[
                [
                    "GeneralizedDiceScore",
                    "MeanIoU",
                ],
            ],
        )
