"""Image resize transforms."""

import functools
from typing import Any, Dict, Sequence

from torchvision import tv_tensors
from torchvision.transforms import v2
from typing_extensions import override

from eva.vision.data.transforms import base
from eva.vision.data.transforms.spatial import functional


class Resize(base.TorchvisionTransformV2):
    """Resize transform for images with spatial or byte-based constraints.

    This transform provides two mutually exclusive modes of resizing:
    1. Spatial resizing: Resize to a specific (height, width) dimension
    2. Byte-based resizing: Resize to fit within a maximum byte size

    The latter is particularly useful for API models (e.g. Claude 3.7) that
    have strict byte size limits for image inputs.
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
                `max_bytes` are provided, `max_bytes` takes precedence.
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
        self.resize_fn = None

        if max_bytes is not None:
            self.resize_fn = functools.partial(functional.resize_to_max_bytes, max_bytes=max_bytes)
        elif size is not None or max_size is not None:
            self.resize_fn = v2.Resize(size=size, max_size=max_size)  # type: ignore

    @functools.singledispatchmethod
    @override
    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return inpt

    @transform.register(tv_tensors.Image)
    @transform.register(tv_tensors.Mask)
    def _(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt_resized = self.resize_fn(inpt) if self.resize_fn is not None else inpt
        return tv_tensors.wrap(inpt_resized, like=inpt)
