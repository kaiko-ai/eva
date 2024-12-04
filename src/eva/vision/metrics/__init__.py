"""Default metric collections API."""

from eva.vision.metrics.defaults.segmentation import MulticlassSegmentationMetrics
from eva.vision.metrics.segmentation.dice import DiceScore
from eva.vision.metrics.segmentation.generalized_dice import GeneralizedDiceScore
from eva.vision.metrics.segmentation.mean_iou import MeanIoU
from eva.vision.metrics.segmentation.monai_dice import MonaiDiceScore

__all__ = [
    "DiceScore",
    "GeneralizedDiceScore",
    "MeanIoU",
    "MonaiDiceScore",
    "MulticlassSegmentationMetrics",
]
