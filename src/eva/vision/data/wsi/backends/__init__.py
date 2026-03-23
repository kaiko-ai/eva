"""WSI Backends API."""

import functools
import importlib.util
from typing import Callable

from eva.vision.data.wsi.backends.base import Wsi

LRU_CACHE_SIZE = 32


def _is_openslide_available() -> bool:
    """Whether the OpenSlide library is available."""
    return importlib.util.find_spec("openslide") is not None


def _is_tiffslide_available() -> bool:
    """Whether the TiffSlide library is available."""
    return importlib.util.find_spec("tiffslide") is not None


def is_backend_available(backend: str) -> bool:
    """Whether the specified backend is available."""
    match backend:
        case "openslide":
            return _is_openslide_available()
        case "tiffslide":
            return _is_tiffslide_available()
    return False


def wsi_backend(backend: str = "openslide") -> Callable[..., Wsi]:
    """Returns the backend to use for reading the whole-slide images."""
    match backend:
        case "openslide":
            if _is_openslide_available():
                from eva.vision.data.wsi.backends.openslide import WsiOpenslide

                return WsiOpenslide
            else:
                raise ValueError(
                    "Missing optional dependency: openslide.\n"
                    "Please install using `pip install openslide-python`."
                )
        case "tiffslide":
            if _is_tiffslide_available():
                from eva.vision.data.wsi.backends.tiffslide import WsiTiffslide

                return WsiTiffslide
            else:
                raise ValueError(
                    "Missing optional dependency: tiffslide.\n"
                    "Please install using `pip install tiffslide`."
                )
        case "pil":
            from eva.vision.data.wsi.backends.pil import PILImage

            return PILImage
        case _:
            raise ValueError(f"Unknown WSI backend selected: {backend}")


@functools.lru_cache(LRU_CACHE_SIZE)
def get_cached_wsi(file_path: str, backend: str, overwrite_mpp: float | None = None) -> Wsi:
    """Returns a cached instance of the whole-slide image backend reader."""
    return wsi_backend(backend)(file_path, overwrite_mpp)


__all__ = ["Wsi", "wsi_backend", "get_cached_wsi", "_is_openslide_available"]
