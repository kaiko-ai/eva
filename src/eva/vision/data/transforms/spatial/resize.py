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
    1. Spatial resizing: Resize to a specific `size` and/or cap the longer
       edge at `max_size`. `max_size` acts exclusively as an upper bound.
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
            max_size: The maximum allowed for the longer edge of the resized
                image. When `size` is None, images whose longer edge is
                already `<= max_size` are returned unchanged without upscaling.

        Raises:
            ValueError: If `max_bytes` is not a positive integer, or if
                `max_size` is provided without `size` on torchvision<0.19.0.
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
            if self._skip_resize(resize_fn, inpt):
                continue
            inpt = resize_fn(inpt)
        return tv_tensors.wrap(inpt, like=inpt)

    def _skip_resize(self, resize_fn: Any, inpt: Any) -> bool:
        """Check if v2.Resize on `inpt` would upscale an image to `max_size` when `size` is None."""
        if self.size is not None or self.max_size is None:
            return False
        if not isinstance(resize_fn, v2.Resize):
            return False
        longest_edge = max(int(inpt.shape[-2]), int(inpt.shape[-1]))
        return longest_edge <= self.max_size
