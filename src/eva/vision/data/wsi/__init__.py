"""WSI API."""

from eva.vision.data.wsi.backends import Wsi, get_cached_wsi, wsi_backend
from eva.vision.data.wsi.patching.coordinates import PatchCoordinates, get_cached_coords

__all__ = ["Wsi", "PatchCoordinates", "get_cached_coords", "wsi_backend", "get_cached_wsi"]
