"""WSI Backends API."""

import functools
import importlib.util
from typing import Callable

from eva.vision.data.wsi.backends.base import Wsi

LRU_CACHE_SIZE = 32


def is_openslide_available() -> bool:
    """Whether the OpenSlide library is available."""
    return importlib.util.find_spec("openslide") is not None


def wsi_backend(backend: str = "openslide") -> Callable[..., Wsi]:
    """Returns the backend to use for reading the whole-slide images."""
    match backend:
        case "openslide":
            if is_openslide_available():
                from eva.vision.data.wsi.backends.openslide import WsiOpenslide

                return WsiOpenslide
            else:
                raise ValueError(
                    "Missing optional dependency: openslide.\n"
                    "Please install using `pip install openslide-python`."
                )
        case _:
            raise ValueError(f"Unknown WSI backend selected: {backend}")


@functools.lru_cache(LRU_CACHE_SIZE)
def get_cached_wsi(file_path: str, backend: str) -> Wsi:
    """Returns a cached instance of the whole-slide image backend reader."""
    return wsi_backend(backend)(file_path)


__all__ = ["Wsi", "wsi_backend", "get_cached_wsi", "is_openslide_available"]
