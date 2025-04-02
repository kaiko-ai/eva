"""Flip transforms."""

import functools
from typing import Any, Dict, List, Sequence

import torch
from monai.transforms.spatial import array as monai_spatial_transforms
from torchvision import tv_tensors
from torchvision.transforms import v2
from typing_extensions import override

from eva.vision.data import tv_tensors as eva_tv_tensors


class RandFlip(v2.Transform):
    """Randomly flips the image along axes."""

    def __init__(
        self,
        prob: float = 0.1,
        spatial_axes: Sequence[int] | int | None = None,
        apply_per_axis: bool = True,
    ) -> None:
        """Initializes the transform.

        Args:
            prob: Probability of flipping.
            spatial_axes: Spatial axes along which to flip over. Default is None.
            apply_per_axis: If True, will apply a random flip transform to each
                axis individually (if spatial_axes is a sequence of multiple axis).
                If False, will apply a single random flip transform applied to all axes.
        """
        super().__init__()

        if apply_per_axis:
            if not isinstance(spatial_axes, (list, tuple)):
                raise ValueError(
                    "`spatial_axis` is expected to be sequence `apply_per_axis` "
                    f"is enabled, got {type(spatial_axes)}"
                )
            self._flips = [
                monai_spatial_transforms.RandFlip(prob=prob, spatial_axis=axis)
                for axis in spatial_axes
            ]
        else:
            self._flips = [monai_spatial_transforms.RandFlip(prob=prob, spatial_axis=spatial_axes)]

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        for flip in self._flips:
            flip.randomize(None)
        return {}

    @functools.singledispatchmethod
    @override
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return inpt

    @_transform.register(tv_tensors.Image)
    @_transform.register(eva_tv_tensors.Volume)
    def _(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt_flipped = self._apply_flips(inpt)
        return tv_tensors.wrap(inpt_flipped, like=inpt)

    @_transform.register(tv_tensors.Mask)
    def _(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt_flipped = torch.tensor(self._apply_flips(inpt), dtype=torch.long)
        return tv_tensors.wrap(inpt_flipped, like=inpt)

    def _apply_flips(self, inpt: Any) -> Any:
        for flip in self._flips:
            inpt = flip(img=inpt, randomize=False)
        return inpt
