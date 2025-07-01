"""Default metric collection for multiclass semantic segmentation tasks."""

from typing import Literal

from eva.core.metrics import structs
from eva.core.utils import requirements
from eva.vision.metrics import segmentation


class MulticlassSegmentationMetrics(structs.MetricCollection):
    """Metrics for multi-class semantic segmentation tasks."""

    def __init__(
        self,
        num_classes: int,
        include_background: bool = False,
        ignore_index: int | None = None,
        prefix: str | None = None,
        postfix: str | None = None,
    ) -> None:
        """Initializes the multi-class semantic segmentation metrics.

        Args:
            num_classes: Integer specifying the number of classes.
            include_background: Whether to include the background class in the metrics computation.
            ignore_index: Integer specifying a target class to ignore. If given, this class
                index does not contribute to the returned score, regardless of reduction method.
            prefix: A string to add before the keys in the output dictionary.
            postfix: A string to add after the keys in the output dictionary.
        """
        super().__init__(
            metrics={
                "MonaiDiceScore": segmentation.MonaiDiceScore(
                    num_classes=num_classes,
                    include_background=include_background,
                    ignore_index=ignore_index,
                    ignore_empty=True,
                ),
                "MonaiDiceScore (ignore_empty=False)": segmentation.MonaiDiceScore(
                    num_classes=num_classes,
                    include_background=include_background,
                    ignore_index=ignore_index,
                    ignore_empty=False,
                ),
                "DiceScore (micro)": segmentation.DiceScore(
                    num_classes=num_classes,
                    include_background=include_background,
                    average="micro",
                    ignore_index=ignore_index,
                ),
                "DiceScore (macro)": segmentation.DiceScore(
                    num_classes=num_classes,
                    include_background=include_background,
                    average="macro",
                    ignore_index=ignore_index,
                ),
                "DiceScore (weighted)": segmentation.DiceScore(
                    num_classes=num_classes,
                    include_background=include_background,
                    average="weighted",
                    ignore_index=ignore_index,
                ),
                "MeanIoU": segmentation.MeanIoU(
                    num_classes=num_classes,
                    include_background=include_background,
                    ignore_index=ignore_index,
                ),
            },
            prefix=prefix,
            postfix=postfix,
        )


class MulticlassSegmentationMetricsV2(structs.MetricCollection):
    """Metrics for multi-class semantic segmentation tasks.

    In torchmetrics 1.8.0, the DiceScore implementation has been
    improved, and should now provide enough signal. Therefore,
    removing the monai implementation and iou for simplicity and
    computational efficiency.
    """

    def __init__(
        self,
        num_classes: int,
        include_background: bool = False,
        prefix: str | None = None,
        postfix: str | None = None,
        input_format: Literal["one-hot", "index"] = "one-hot",
    ) -> None:
        """Initializes the multi-class semantic segmentation metrics.

        Args:
            num_classes: Integer specifying the number of classes.
            include_background: Whether to include the background class in the metrics computation.
            prefix: A string to add before the keys in the output dictionary.
            postfix: A string to add after the keys in the output dictionary.
            input_format: Input tensor format. Options are `"one-hot"` for one-hot encoded tensors,
                `"index"` for index tensors.
        """
        requirements.check_dependencies(requirements={"torchmetrics": "1.8.0"})
        super().__init__(
            metrics={
                "DiceScore (macro)": segmentation.DiceScore(
                    num_classes=num_classes,
                    include_background=include_background,
                    average="macro",
                    aggregation_level="samplewise",
                    input_format=input_format,
                ),
                "DiceScore (macro/global)": segmentation.DiceScore(
                    num_classes=num_classes,
                    include_background=include_background,
                    average="macro",
                    aggregation_level="global",
                    input_format=input_format,
                ),
            },
            prefix=prefix,
            postfix=postfix,
        )
        self.num_classes = num_classes
