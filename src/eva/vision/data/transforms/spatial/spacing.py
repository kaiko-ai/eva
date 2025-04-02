"""Spacing resample transform."""

import functools
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from monai.data import meta_tensor
from monai.transforms.spatial import array as monai_spatial_transforms
from torchvision import tv_tensors
from torchvision.transforms import v2
from typing_extensions import override

from eva.vision.data import tv_tensors as eva_tv_tensors


class Spacing(v2.Transform):
    """Resample input image into the specified `pixdim`.

    - Expects tensors of shape `[C, T, H, W]`.
    """

    def __init__(
        self,
        pixdim: Sequence[float] | float | np.ndarray,
    ) -> None:
        """Initializes the transform.

        Args:
            pixdim: output voxel spacing. if providing a single number,
                will use it for the first dimension. Items of the pixdim
                sequence map to the spatial dimensions of input image, if
                length of pixdim sequence is longer than image spatial
                dimensions, will ignore the longer part, if shorter, will
                pad with the last value. For example, for 3D image if pixdim
                is [1.0, 2.0] it will be padded to [1.0, 2.0, 2.0] if the
                components of the `pixdim` are non-positive values, the
                transform will use the corresponding components of the original
                pixdim, which is computed from the `affine` matrix of input image.
        """
        super().__init__()

        self._spacing = monai_spatial_transforms.Spacing(pixdim=pixdim, recompute_affine=True)
        self._affine = None

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        self._affine = next(
            inpt.affine for inpt in flat_inputs if isinstance(inpt, eva_tv_tensors.Volume)
        )
        return {}

    @functools.singledispatchmethod
    @override
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return inpt

    @_transform.register(eva_tv_tensors.Volume)
    def _(self, inpt: eva_tv_tensors.Volume, params: Dict[str, Any]) -> Any:
        inpt_spacing = self._spacing(inpt.to_meta_tensor(), mode="bilinear")
        if not isinstance(inpt_spacing, meta_tensor.MetaTensor):
            raise ValueError(f"Expected MetaTensor, got {type(inpt_spacing)}")
        return eva_tv_tensors.Volume.from_meta_tensor(inpt_spacing)

    @_transform.register(tv_tensors.Mask)
    def _(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt_spacing = self._spacing(
            meta_tensor.MetaTensor(inpt, affine=self._affine), mode="nearest"
        )
        return tv_tensors.wrap(inpt_spacing.to(dtype=torch.long), like=inpt)
