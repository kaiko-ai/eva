"""Intensity scaling transform."""

import functools
from typing import Any, Dict, Tuple

from monai.transforms.intensity import array as monai_intensity_transforms
from torchvision import tv_tensors
from torchvision.transforms import v2
from typing_extensions import override

from eva.vision.data import tv_tensors as eva_tv_tensors


class ScaleIntensityRange(v2.Transform):
    """Intensity scaling transform.

    Scaling from [a_min, a_max] to [b_min, b_max] with clip option.

    When `b_min` or `b_max` are `None`, `scaled_array * (b_max - b_min) + b_min`
    will be skipped. If `clip=True`, when `b_min`/`b_max` is None, the clipping
    is not performed on the corresponding edge.
    """

    def __init__(
        self,
        input_range: Tuple[float, float],
        output_range: Tuple[float, float] | None = None,
        clip: bool = True,
    ) -> None:
        """Initializes the transform.

        Args:
            input_range: Intensity original range min and max.
            output_range: Intensity target range min and max.
            clip: Whether to perform clip after scaling.
        """
        super().__init__()

        self._scale_intensity_range = monai_intensity_transforms.ScaleIntensityRange(
            a_min=input_range[0],
            a_max=input_range[1],
            b_min=output_range[0] if output_range else None,
            b_max=output_range[1] if output_range else None,
            clip=clip,
        )

    @functools.singledispatchmethod
    @override
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return inpt

    @_transform.register(tv_tensors.Image)
    @_transform.register(eva_tv_tensors.Volume)
    def _(self, inpt: tv_tensors.Image, params: Dict[str, Any]) -> Any:
        inpt_scaled = self._scale_intensity_range(inpt)
        return tv_tensors.wrap(inpt_scaled, like=inpt)
