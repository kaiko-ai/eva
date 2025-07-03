"""Crop foreground transform."""

import functools
from typing import Any, Dict, List, Sequence

import torch
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.croppad import array as monai_croppad_transforms
from torchvision import tv_tensors
from typing_extensions import override

from eva.vision.data import tv_tensors as eva_tv_tensors
from eva.vision.data.transforms import base


class RandCropByPosNegLabel(base.RandomMonaiTransform):
    """Crop random fixed sized regions with the center being a foreground or background voxel.

    Its based on the Pos Neg Ratio and will return a list of arrays for all the cropped images.
    For example, crop two (3 x 3) arrays from (5 x 5) array with pos/neg=1::

        [[[0, 0, 0, 0, 0],
          [0, 1, 2, 1, 0],            [[0, 1, 2],     [[2, 1, 0],
          [0, 1, 3, 0, 0],     -->     [0, 1, 3],      [3, 0, 0],
          [0, 0, 0, 0, 0],             [0, 0, 0]]      [0, 0, 0]]
          [0, 0, 0, 0, 0]]]

    If a dimension of the expected spatial size is larger than the input image size,
    will not crop that dimension. So the cropped result may be smaller than expected
    size, and the cropped results of several images may not have exactly same shape.
    """

    def __init__(
        self,
        spatial_size: Sequence[int] | int,
        label: torch.Tensor | None = None,
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 1,
        image: torch.Tensor | None = None,
        image_threshold: float = 0.0,
        fg_indices: NdarrayOrTensor | None = None,
        bg_indices: NdarrayOrTensor | None = None,
        allow_smaller: bool = False,
    ) -> None:
        """Initializes the transform.

        Args:
            spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
                if a dimension of ROI size is larger than image size, will not crop that dimension.
                if components have non-positive values, corresponding size of `label` will be used.
                for example: if the spatial size of input data is [40, 40, 40] and
                `spatial_size=[32, 64, -1]`, the spatial size of output data will be [32, 40, 40].
            label: the label image that is used for finding foreground/background, if None, must
                set at `self.__call__`. Non-zero indicates foreground, zero indicates background.
            pos: used with `neg` together to calculate the ratio ``pos / (pos + neg)`` for
                the probability to pick a foreground voxel as center rather than background voxel.
            neg: used with `pos` together to calculate the ratio ``pos / (pos + neg)`` for
                the probability to pick a foreground voxel as center rather than background voxel.
            num_samples: number of samples (crop regions) to take in each list.
            image: optional image data to help select valid area, can be same as `img` or another.
                if not None, use ``label == 0 & image > image_threshold`` to select the negative
                sample (background) center. Crop center will only come from valid image areas.
            image_threshold: if enabled `image`, use ``image > image_threshold`` to determine
                the valid image content areas.
            fg_indices: if provided pre-computed foreground indices of `label`, will ignore `image`
                and `image_threshold`, randomly select crop centers based on them, need to provide
                `fg_indices` and `bg_indices` together, expect to be 1 dim array of spatial indices.
                a typical usage is to call `FgBgToIndices` transform first and cache the results.
            bg_indices: if provided pre-computed background indices of `label`, will ignore `image`
                and `image_threshold`, randomly select crop centers based on them, need to provide
                `fg_indices` and `bg_indices` together, expect to be 1 dim array of spatial indices.
                a typical usage is to call `FgBgToIndices` transform first and cache the results.
            allow_smaller: if `False`, an exception will be raised if the image is smaller than
                the requested ROI in any dimension. If `True`, any smaller dimensions will be set to
                match the cropped size (i.e., no cropping in that dimension).
        """
        super().__init__()

        self._rand_crop = monai_croppad_transforms.RandCropByPosNegLabel(
            spatial_size=spatial_size,
            label=label,
            pos=pos,
            neg=neg,
            num_samples=num_samples,
            image=image,
            image_threshold=image_threshold,
            fg_indices=fg_indices,
            bg_indices=bg_indices,
            allow_smaller=allow_smaller,
            lazy=False,
        )

    @override
    def set_random_state(self, seed: int) -> None:
        self._rand_crop.set_random_state(seed)

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        mask = next(inpt for inpt in flat_inputs if isinstance(inpt, tv_tensors.Mask))
        self._rand_crop.randomize(label=mask)
        return {}

    @functools.singledispatchmethod
    @override
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return inpt

    @_transform.register(tv_tensors.Image)
    @_transform.register(eva_tv_tensors.Volume)
    @_transform.register(tv_tensors.Mask)
    def _(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt_foreground_crops = self._rand_crop(img=inpt, randomize=False)
        return [tv_tensors.wrap(crop, like=inpt) for crop in inpt_foreground_crops]
