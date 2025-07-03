"""Intensity shifting transform."""

import functools
from typing import Any, Dict

from monai.transforms.intensity import array as monai_intensity_transforms
from torchvision import tv_tensors
from typing_extensions import override

from eva.vision.data import tv_tensors as eva_tv_tensors
from eva.vision.data.transforms import base


class RandShiftIntensity(base.RandomMonaiTransform):
    """Randomly shift intensity with randomly picked offset."""

    def __init__(
        self,
        offsets: tuple[float, float] | float,
        safe: bool = False,
        prob: float = 0.1,
        channel_wise: bool = False,
    ) -> None:
        """Initializes the transform.

        Args:
            offsets: Offset range to randomly shift.
                if single number, offset value is picked from (-offsets, offsets).
            safe: If `True`, then do safe dtype convert when intensity overflow.
                E.g., `[256, -12]` -> `[array(0), array(244)]`. If `True`, then
                `[256, -12]` -> `[array(255), array(0)]`.
            prob: Probability of shift.
            channel_wise: If True, shift intensity on each channel separately.
                For each channel, a random offset will be chosen. Please ensure
                that the first dimension represents the channel of the image if True.
        """
        super().__init__()

        self._rand_shift_intensity = monai_intensity_transforms.RandShiftIntensity(
            offsets=offsets,
            safe=safe,
            prob=prob,
            channel_wise=channel_wise,
        )

    @override
    def set_random_state(self, seed: int) -> None:
        self._rand_shift_intensity.set_random_state(seed)

    @functools.singledispatchmethod
    @override
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return inpt

    @_transform.register(tv_tensors.Image)
    @_transform.register(eva_tv_tensors.Volume)
    def _(self, inpt: tv_tensors.Image, params: Dict[str, Any]) -> Any:
        inpt_scaled = self._rand_shift_intensity(inpt)
        return tv_tensors.wrap(inpt_scaled, like=inpt)
