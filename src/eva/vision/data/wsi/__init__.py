from eva.vision.data.wsi.backend import WsiBackend, get_wsi_class
from eva.vision.data.wsi.base import Wsi
from eva.vision.data.wsi.openslide import WsiOpenslide

__all__ = ["Wsi", "WsiOpenslide", "WsiBackend", "get_wsi_class"]
