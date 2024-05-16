from typing import Any, Dict

from torchvision.transforms import v2


class RescaleIntensity(v2.Transform):
    """Return image after stretching or shrinking its intensity levels.

    The desired intensity range of the input and output, `in_range` and
    `out_range` respectively, are used to stretch or shrink the intensity
    range of the input image. See examples below.


    Scales the data between 0..1 based on a window with lower and upper
    limits as specified. dtype must be a float type.
    """

    def __init__(self, lower: float = -125., upper: float = 225.) -> None:
        super().__init__()

        self._lower = lower
        self._upper = upper


    def _transform(self, data: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(F.rgb_to_grayscale, data, lower=self._lower, upper=self._upper)
