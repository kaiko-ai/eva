"""Crop image with random size or specific size ROI."""

import functools
from typing import Any, Dict, List, Sequence, Tuple

from monai.transforms.croppad import array as monai_croppad_transforms
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import _utils as tv_utils
from typing_extensions import override

from eva.vision.data import tv_tensors as eva_tv_tensors


class RandSpatialCrop(v2.Transform):
    """Crop image with random size or specific size ROI.

    It can crop at a random position as center or at the image center.
    And allows to set the minimum and maximum size to limit the randomly
    generated ROI.
    """

    def __init__(
        self,
        roi_size: Sequence[int] | int,
        max_roi_size: Sequence[int] | int | None = None,
        random_center: bool = True,
        random_size: bool = False,
    ) -> None:
        """Initializes the transform.

        Args:
            roi_size: if `random_size` is True, it specifies the minimum crop
                region. if `random_size` is False, it specifies the expected
                ROI size to crop. e.g. [224, 224, 128]. if a dimension of ROI
                size is larger than image size, will not crop that dimension of
                the image. If its components have non-positive values, the
                corresponding size of input image will be used. for example: if
                the spatial size of input data is [40, 40, 40] and
                `roi_size=[32, 64, -1]`, the spatial size of output data will be
                [32, 40, 40].
            max_roi_size: if `random_size` is True and `roi_size` specifies the
                min crop region size, `max_roi_size` can specify the max crop
                region size. if None, defaults to the input image size. if its
                components have non-positive values, the corresponding size of
                input image will be used.
            random_center: crop at random position as center or the image center.
            random_size: crop with random size or specific size ROI. if True, the
                actual size is sampled from `randint(roi_size, max_roi_size + 1)`.
        """
        super().__init__()

        self._rand_spatial_crop = monai_croppad_transforms.RandSpatialCrop(
            roi_size=roi_size,
            max_roi_size=max_roi_size,
            random_center=random_center,
            random_size=random_size,
        )
        self._cropper = monai_croppad_transforms.Crop()

    def set_random_state(self, seed: int) -> None:
        """Set the random state for the transform."""
        self._rand_spatial_crop.set_random_state(seed)

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        t, h, w = tv_utils.query_chw(flat_inputs)
        self._rand_spatial_crop.randomize((t, h, w))
        return {}

    @functools.singledispatchmethod
    @override
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return inpt

    @_transform.register(tv_tensors.Image)
    @_transform.register(eva_tv_tensors.Volume)
    @_transform.register(tv_tensors.Mask)
    def _(self, inpt: Any, params: Dict[str, Any]) -> Any:
        slices = self._get_crop_slices()
        inpt_rand_crop = self._cropper(inpt, slices=slices)
        return tv_tensors.wrap(inpt_rand_crop, like=inpt)

    def _get_crop_slices(self) -> Tuple[slice, ...]:
        """Returns the sequence of slices to crop."""
        if self._rand_spatial_crop.random_center:
            return self._rand_spatial_crop._slices

        central_cropper = monai_croppad_transforms.CenterSpatialCrop(self._size)
        return central_cropper.compute_slices(self._rand_spatial_crop._size)  # type: ignore
