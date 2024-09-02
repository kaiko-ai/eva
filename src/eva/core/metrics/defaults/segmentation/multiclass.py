"""Default metric collection for multiclass semantic segmentation tasks."""

from typing import Literal

from eva.core.metrics import generalized_dice, mean_iou, structs


class MulticlassSegmentationMetrics(structs.MetricCollection):
    """Default metrics for multi-class semantic segmentation tasks."""

    def __init__(
        self,
        num_classes: int,
        include_background: bool = False,
        input_format: Literal["one-hot", "index"] = "index",
        ignore_index: int | None = None,
        prefix: str | None = None,
        postfix: str | None = None,
    ) -> None:
        """Initializes the multi-class semantic segmentation metrics.

        Args:
            num_classes: Integer specifying the number of classes.
            include_background: Whether to include the background class in the metrics computation.
            input_format: What kind of input the metrics should expect. With `"one-hot"` it would
                expect a stack of binary masks, one for each class and for with `"index"` a 2D mask
                where each pixel is annotated with the class ID.
            ignore_index: Integer specifying a target class to ignore. If given, this class
                index does not contribute to the returned score, regardless of reduction method.
            prefix: A string to add before the keys in the output dictionary.
            postfix: A string to add after the keys in the output dictionary.
        """
        super().__init__(
            metrics=[
                generalized_dice.GeneralizedDiceScore(
                    num_classes=num_classes,
                    include_background=include_background,
                    input_format=input_format,
                    weight_type="linear",
                    ignore_index=ignore_index,
                ),
                mean_iou.MeanIoU(
                    num_classes=num_classes,
                    include_background=include_background,
                    input_format=input_format,
                    ignore_index=ignore_index,
                ),
            ],
            prefix=prefix,
            postfix=postfix,
        )
