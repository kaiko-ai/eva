"""Image resize transforms."""

import functools
from typing import Any, Dict, Sequence

from torchvision import tv_tensors
from torchvision.transforms import v2
from typing_extensions import override

from eva.core.utils import requirements
from eva.vision.data.transforms import base
from eva.vision.data.transforms.spatial import functional


class Resize(base.TorchvisionTransformV2):
    """Resize transform for images.

    This transform provides different modes of resizing:
    1. Spatial resizing: Resize to a specific size dimension or
       maximum size for the longer edge.
    2. Byte-based resizing: Resize to fit within a maximum byte size.

    If both spatial and byte-based constraints are provided, first the spatial
    resizing is applied, followed by byte-based resizing. The reasoning behind
    this is that byte-based resizing can be slow for large images, so first
    applying the spatial resizing, and then applying byte-based (e.g. needed to
    meet API image size limits) can be more efficient.
    """

    def __init__(
        self,
        size: int | Sequence[int] | None = None,
        max_bytes: int | None = None,
        max_size: int | None = None,
    ) -> None:
        """Initializes the transform.

        Args:
            size: Desired output size, e.g. (height, width) tuple.
            max_bytes: Maximum allowed byte size for the image. If both `size` and
                `max_bytes` are provided, spatial resizing is applied first.
            max_size: The maximum allowed for the longer edge of the resized image.

        Raises:
            ValueError: If both size and max_bytes are provided, or if max_bytes
                is not a positive integer.
        """
        if max_bytes is not None and max_bytes <= 0:
            raise ValueError("'max_bytes' must be a positive integer.")

        super().__init__()

        self.size = size
        self.max_bytes = max_bytes
        self.max_size = max_size
        self.resize_fns = []

        if size is not None or max_size is not None:
            if requirements.below("torchvision", "0.19.0") and (
                size is None and max_size is not None
            ):
                raise ValueError(
                    "Setting `max_size` without `size` is only supported in torchvision>=0.19.0."
                )
            self.resize_fns.append(v2.Resize(size=size, max_size=max_size))  # type: ignore
        if max_bytes is not None:
            self.resize_fns.append(
                functools.partial(functional.resize_to_max_bytes, max_bytes=max_bytes)
            )

    @functools.singledispatchmethod
    @override
    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return inpt

    @transform.register(tv_tensors.Image)
    @transform.register(tv_tensors.Mask)
    def _(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if not self.resize_fns:
            return inpt
        for resize_fn in self.resize_fns:
            inpt = resize_fn(inpt)
        return tv_tensors.wrap(inpt, like=inpt)
