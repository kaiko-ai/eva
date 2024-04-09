import enum

from eva.vision.data import wsi


class WsiBackend(enum.Enum):
    OPENSLIDE = (0,)
    AUTO = 1


def get_wsi_class(backend: WsiBackend) -> wsi.Wsi:
    match backend:
        case WsiBackend.OPENSLIDE:
            return wsi.WsiOpenslide
        case WsiBackend.AUTO:
            raise NotImplementedError
        case _:
            raise ValueError(f"Unknown WSI backend: {backend}")
