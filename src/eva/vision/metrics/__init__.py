"""Default metric collections API."""

from eva.vision.metrics.defaults.segmentation import MulticlassSegmentationMetrics
from eva.vision.metrics.segmentation.generalized_dice import GeneralizedDiceScore
from eva.vision.metrics.segmentation.mean_iou import MeanIoU

__all__ = [
    "MulticlassSegmentationMetrics",
    "GeneralizedDiceScore",
    "MeanIoU",
]
