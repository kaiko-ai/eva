"""Rotation transforms."""

import functools
from typing import Any, Dict, List

from monai.transforms.spatial import array as monai_spatial_transforms
from torchvision import tv_tensors
from torchvision.transforms import v2
from typing_extensions import override

from eva.vision.data import tv_tensors as eva_tv_tensors


class RandRotate90(v2.Transform):
    """Rotate input tensors by 90 degrees."""

    def __init__(
        self,
        prob: float = 0.1,
        max_k: int = 3,
        spatial_axes: tuple[int, int] = (1, 2),
    ) -> None:
        """Initializes the transform.

        Args:
            prob: probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array)
            max_k: number of rotations will be sampled from `np.random.randint(max_k) + 1`.
            spatial_axes: 2 int numbers, defines the plane to rotate with 2 spatial axes.
                Default: (1, 2), so for [C, T, H, W] will rotate along (H, W) plane (MONAI ignores
                the first C dimension).
        """
        super().__init__()

        self._rotate = monai_spatial_transforms.RandRotate90(
            prob=prob, max_k=max_k, spatial_axes=spatial_axes
        )

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        self._rotate.randomize()
        return {}

    @functools.singledispatchmethod
    @override
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return inpt

    @_transform.register(tv_tensors.Image)
    @_transform.register(eva_tv_tensors.Volume)
    @_transform.register(tv_tensors.Mask)
    def _(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt_rotated = self._rotate(img=inpt, randomize=False)
        return tv_tensors.wrap(inpt_rotated, like=inpt)
