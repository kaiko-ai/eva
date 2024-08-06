"""Loss functions API."""

from eva.vision.losses.dice import DiceLoss
from eva.vision.losses.mask2former import Mask2FormerLoss

__all__ = ["DiceLoss", "Mask2FormerLoss"]
