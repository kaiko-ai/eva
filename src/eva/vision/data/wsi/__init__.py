"""WSI API."""

from eva.vision.data.wsi.backends import wsi_backend
from eva.vision.data.wsi.base import Wsi
from eva.vision.data.wsi.patching.coordinates import PatchCoordinates

__all__ = ["Wsi", "PatchCoordinates", "wsi_backend"]
