"""Loss functions API."""

from eva.vision.losses.dice import DiceLoss
from eva.vision.losses.mask2former import Mask2formerLoss

__all__ = ["DiceLoss", "Mask2formerLoss"]
