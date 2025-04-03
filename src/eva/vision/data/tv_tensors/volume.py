"""Custom `tv_tensors` type for 3D Volumes."""

from typing import Any, Dict, Optional, Union

import torch
from monai.data import meta_tensor
from torchvision import tv_tensors
from typing_extensions import override


class Volume(tv_tensors.Video):
    """:class:`torchvision.TVTensor` subclass for 3D volumes.

    - Adds optional metadata and affine matrix to the tensor.
    - Expects tensors to be of shape `[..., T, C, H, W]`.

    Args:
        data: Any data that can be turned into a tensor with :func:`torch.as_tensor`.
        affine: Affine matrix of the volume. Expected to be of shape `[4, 4]`, and
            columns to correspond to [T, H, W, (translation)] dimensions. Note that
            `nibabel` by default uses [H, W, T, (translation)] order for affine matrices.
        metadata: Metadata associated with the volume.
        dtype: Desired data type. If omitted, will be inferred from `data`.
        device: Desired device.
        requires_grad: Whether autograd should record operations.
    """

    @override
    def __new__(
        cls,
        data: Any,
        affine: torch.Tensor | None = None,
        metadata: Dict[str, Any] | None = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> "Volume":
        cls.affine = affine
        cls.metadata = metadata

        return super().__new__(cls, data, dtype=dtype, device=device, requires_grad=requires_grad)  # type: ignore

    @classmethod
    def from_meta_tensor(cls, meta_tensor: meta_tensor.MetaTensor) -> "Volume":
        """Creates an instance from a :class:`monai.data.meta_tensor.MetaTensor`."""
        return cls(
            meta_tensor.data,
            affine=meta_tensor.affine,
            metadata=meta_tensor.meta,
            dtype=meta_tensor.dtype,
            device=meta_tensor.device,
            requires_grad=meta_tensor.requires_grad,
        )  # type: ignore

    def to_meta_tensor(self) -> meta_tensor.MetaTensor:
        """Converts the volume to a :class:`monai.data.meta_tensor.MetaTensor`."""
        return meta_tensor.MetaTensor(self, affine=self.affine, meta=self.metadata)

    def __repr__(self, *, tensor_contents: Any = None) -> str:
        """Returns the string representation of the object."""
        return self._make_repr()
