"""Module for loading different WSI backends."""

import enum
from typing import Callable

from eva.vision.data.wsi.base import Wsi
from eva.vision.data.wsi.openslide import WsiOpenslide


class WsiBackend(enum.Enum):
    """Supported WSI backends."""

    OPENSLIDE = 0
    AUTO = 1


def get_wsi_class(backend: WsiBackend) -> Callable[..., Wsi]:
    """Returns the WSI class for the given backend.

    Args:
        backend: The backend to use for reading the whole-slide images.
    """
    match backend:
        case WsiBackend.OPENSLIDE:
            return WsiOpenslide
        case WsiBackend.AUTO:
            raise NotImplementedError
        case _:
            raise ValueError(f"Unknown WSI backend: {backend}")
