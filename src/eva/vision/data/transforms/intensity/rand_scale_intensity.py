"""Intensity scaling transform."""

import functools
from typing import Any, Dict

import numpy as np
from monai.config.type_definitions import DtypeLike
from monai.transforms.intensity import array as monai_intensity_transforms
from torchvision import tv_tensors
from typing_extensions import override

from eva.vision.data import tv_tensors as eva_tv_tensors
from eva.vision.data.transforms import base


class RandScaleIntensity(base.RandomMonaiTransform):
    """Randomly scale the intensity of input image.

    The factor is by ``v = v * (1 + factor)``, where
    the `factor` is randomly picked.
    """

    def __init__(
        self,
        factors: tuple[float, float] | float,
        prob: float = 0.1,
        channel_wise: bool = False,
        dtype: DtypeLike = np.float32,
    ) -> None:
        """Initializes the transform.

        Args:
            factors: factor range to randomly scale by ``v = v * (1 + factor)``.
                if single number, factor value is picked from (-factors, factors).
            prob: probability of scale.
            channel_wise: if True, shift intensity on each channel separately.
                For each channel, a random offset will be chosen. Please ensure
                that the first dimension represents the channel of the image if True.
            dtype: output data type, if None, same as input image. defaults to float32.
        """
        super().__init__()

        self._rand_scale_intensity = monai_intensity_transforms.RandScaleIntensity(
            factors=factors,
            prob=prob,
            channel_wise=channel_wise,
            dtype=dtype,
        )

    @override
    def set_random_state(self, seed: int) -> None:
        self._rand_scale_intensity.set_random_state(seed)

    @functools.singledispatchmethod
    @override
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return inpt

    @_transform.register(tv_tensors.Image)
    @_transform.register(eva_tv_tensors.Volume)
    def _(self, inpt: tv_tensors.Image, params: Dict[str, Any]) -> Any:
        inpt_scaled = self._rand_scale_intensity(inpt)
        return tv_tensors.wrap(inpt_scaled, like=inpt)
