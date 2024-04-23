"""WSI API."""

from eva.vision.data.wsi.backends import Wsi, wsi_backend
from eva.vision.data.wsi.patching.coordinates import PatchCoordinates

__all__ = ["Wsi", "PatchCoordinates", "wsi_backend"]
