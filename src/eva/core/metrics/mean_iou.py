"""Mean Intersection over Union (mIoU) metric for semantic segmentation."""

from typing import Any, Literal, Tuple

import torch
import torchmetrics


class MeanIoU(torchmetrics.Metric):
    """Computes Mean Intersection over Union (mIoU) for semantic segmentation.

    Fixes the torchmetrics implementation
    (issue https://github.com/Lightning-AI/torchmetrics/issues/2558)
    """

    def __init__(
        self,
        num_classes: int,
        include_background: bool = True,
        ignore_index: int | None = None,
        per_class: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initializes the metric.

        Args:
            num_classes: The number of classes in the segmentation problem.
            include_background: Whether to include the background class in the computation
            ignore_index: Integer specifying a target class to ignore. If given, this class
                index does not contribute to the returned score, regardless of reduction method.
            per_class: Whether to compute the IoU for each class separately. If set to ``False``,
                the metric will compute the mean IoU over all classes.
            kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.
        """
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.include_background = include_background
        self.ignore_index = ignore_index
        self.per_class = per_class

        self.add_state("intersection", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("union", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update the state with the new data."""
        intersection, union = _compute_intersection_and_union(
            preds,
            target,
            num_classes=self.num_classes,
            include_background=self.include_background,
            ignore_index=self.ignore_index,
        )
        self.intersection += intersection.sum(0)
        self.union += union.sum(0)

    def compute(self) -> torch.Tensor:
        """Compute the final mean IoU score."""
        iou_valid = torch.gt(self.union, 0)
        iou = torch.where(
            iou_valid,
            torch.divide(self.intersection, self.union),
            torch.nan,
        )
        if not self.per_class:
            iou = torch.mean(iou[iou_valid])
        return iou


def _compute_intersection_and_union(
    preds: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    include_background: bool = False,
    input_format: Literal["one-hot", "index"] = "index",
    ignore_index: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the intersection and union for semantic segmentation tasks.

    Args:
        preds: Predicted tensor with shape (N, ...) where N is the batch size.
            The shape can be (N, H, W) for 2D data or (N, D, H, W) for 3D data.
        target: Ground truth tensor with the same shape as preds.
        num_classes: Number of classes in the segmentation task.
        include_background: Whether to include the background class in the computation.
        input_format: Format of the input tensors.
        ignore_index: Integer specifying a target class to ignore. If given, this class
            index does not contribute to the returned score, regardless of reduction method.

    Returns:
        Two tensors representing the intersection and union for each class.
        Shape of each tensor is (N, num_classes).

    Note:
        - If input_format is "index", the tensors are converted to one-hot encoding.
        - If include_background is `False`, the background class
          (assumed to be the first channel) is ignored in the computation.
    """
    if ignore_index is not None:
        mask = target != ignore_index
        mask = mask.all(dim=-1, keepdim=True)
        preds = preds * mask
        target = target * mask

    if input_format == "index":
        preds = torch.nn.functional.one_hot(preds, num_classes=num_classes)
        target = torch.nn.functional.one_hot(target, num_classes=num_classes)

    if not include_background:
        preds[..., 0] = 0
        target[..., 0] = 0

    reduce_axis = list(range(1, preds.ndim - 1))

    intersection = torch.sum(torch.logical_and(preds, target), dim=reduce_axis)
    target_sum = torch.sum(target, dim=reduce_axis)
    pred_sum = torch.sum(preds, dim=reduce_axis)
    union = target_sum + pred_sum - intersection

    return intersection, union
