"""Segmentation metrics API."""

from eva.vision.metrics.segmentation.dice import DiceScore
from eva.vision.metrics.segmentation.generalized_dice import GeneralizedDiceScore
from eva.vision.metrics.segmentation.mean_iou import MeanIoU

__all__ = [
    "DiceScore",
    "GeneralizedDiceScore",
    "MeanIoU",
]
