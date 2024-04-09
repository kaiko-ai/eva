import enum

from eva.vision.data.wsi.base import Wsi
from eva.vision.data.wsi.openslide import WsiOpenslide


class WsiBackend(enum.Enum):
    OPENSLIDE = 0
    AUTO = 1


def get_wsi_class(backend: WsiBackend) -> Wsi:
    match backend:
        case WsiBackend.OPENSLIDE:
            return WsiOpenslide
        case WsiBackend.AUTO:
            raise NotImplementedError
        case _:
            raise ValueError(f"Unknown WSI backend: {backend}")
