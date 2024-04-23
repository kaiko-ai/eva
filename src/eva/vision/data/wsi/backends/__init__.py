"""WSI Backends API."""

import importlib.util
from typing import Callable

from eva.vision.data.wsi import base


def is_openslide_available() -> bool:
    """Whether the OpenSlide library is available."""
    return importlib.util.find_spec("openslide") is not None


def wsi_backend(backend: str = "openslide") -> Callable[..., base.Wsi]:
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


__all__ = ["is_openslide_available", "wsi_backend"]
