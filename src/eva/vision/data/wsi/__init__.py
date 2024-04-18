"""WSI API."""

from eva.vision.data.wsi.backend import WsiBackend, get_wsi_class
from eva.vision.data.wsi.base import Wsi
from eva.vision.data.wsi.openslide import WsiOpenslide
from eva.vision.data.wsi.patching.coordinates import PatchCoordinates

__all__ = ["Wsi", "WsiOpenslide", "WsiBackend", "PatchCoordinates", "get_wsi_class"]
