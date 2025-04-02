"""General purpose cropper to produce sub-volume region of interest (ROI)."""

import functools
from typing import Any, Dict, Sequence

from monai.transforms.croppad import array as monai_croppad_transforms
from monai.utils.enums import Method, PytorchPadMode
from torchvision import tv_tensors
from torchvision.transforms import v2
from typing_extensions import override

from eva.vision.data import tv_tensors as eva_tv_tensors


class SpatialPad(v2.Transform):
    """Performs padding to the data.

    Padding is applied symmetric for all sides or all on one side for each dimension.
    """

    def __init__(
        self,
        spatial_size: Sequence[int] | int | tuple[tuple[int, ...] | int, ...],
        method: str = Method.SYMMETRIC,
        mode: str = PytorchPadMode.CONSTANT,
    ) -> None:
        """Initializes the transform.

        Args:
            spatial_size: The spatial size of output data after padding.
                If a dimension of the input data size is larger than the
                pad size, will not pad that dimension. If its components
                have non-positive values, the corresponding size of input
                image will be used (no padding). for example: if the spatial
                size of input data is [30, 30, 30] and `spatial_size=[32, 25, -1]`,
                the spatial size of output data will be [32, 30, 30].
            method: {``"symmetric"``, ``"end"``}
                Pad image symmetrically on every side or only pad at the
                end sides. Defaults to ``"symmetric"``.
            mode: available modes for numpy array:{``"constant"``, ``"edge"``,
                ``"linear_ramp"``, ``"maximum"``, ``"mean"``, ``"median"``, ``"minimum"``,
                ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``,
                ``"circular"``}. One of the listed string values or a user supplied function.
                Defaults to ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        """
        super().__init__()

        self._spatial_pad = monai_croppad_transforms.SpatialPad(
            spatial_size=spatial_size,
            method=method,
            mode=mode,
        )

    @functools.singledispatchmethod
    @override
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return inpt

    @_transform.register(tv_tensors.Image)
    @_transform.register(eva_tv_tensors.Volume)
    @_transform.register(tv_tensors.Mask)
    def _(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt_padded = self._spatial_pad(inpt)
        return tv_tensors.wrap(inpt_padded, like=inpt)
