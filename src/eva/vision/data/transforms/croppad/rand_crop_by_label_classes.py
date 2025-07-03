"""Crop by label classes transform."""

import functools
from typing import Any, Dict, List, Sequence

import torch
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.croppad import array as monai_croppad_transforms
from torchvision import tv_tensors
from typing_extensions import override

from eva.vision.data import tv_tensors as eva_tv_tensors
from eva.vision.data.transforms import base


class RandCropByLabelClasses(base.RandomMonaiTransform):
    """Crop random fixed sized regions with the center belonging to one of the classes.

    Please refer to `monai.transforms.croppad.RandCropByLabelClasses` docs for more details.
    """

    def __init__(
        self,
        spatial_size: Sequence[int] | int,
        ratios: list[float | int] | None = None,
        label: torch.Tensor | None = None,
        num_classes: int | None = None,
        num_samples: int = 1,
        image: torch.Tensor | None = None,
        image_threshold: float = 0.0,
        indices: list[NdarrayOrTensor] | None = None,
        allow_smaller: bool = False,
        warn: bool = True,
        max_samples_per_class: int | None = None,
        lazy: bool = False,
    ) -> None:
        """Initializes the transform."""
        super().__init__()

        self._rand_crop = monai_croppad_transforms.RandCropByLabelClasses(
            spatial_size=spatial_size,
            ratios=ratios,
            label=label,
            num_classes=num_classes,
            num_samples=num_samples,
            image=image,
            image_threshold=image_threshold,
            indices=indices,
            allow_smaller=allow_smaller,
            warn=warn,
            max_samples_per_class=max_samples_per_class,
            lazy=lazy,
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
