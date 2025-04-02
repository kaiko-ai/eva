"""Adjust or add the channel dimension of input data to ensure `channel_first` shape."""

import functools
from typing import Any, Dict

from monai.transforms.utility import array as monai_utility_transforms
from torchvision import tv_tensors
from torchvision.transforms import v2
from typing_extensions import override

from eva.vision.data import tv_tensors as eva_tv_tensors


class EnsureChannelFirst(v2.Transform):
    """Adjust or add the channel dimension of input data to ensure `channel_first` shape."""

    def __init__(
        self,
        strict_check: bool = True,
        channel_dim: None | str | int = None,
    ) -> None:
        """Initializes the transform.

        Args:
            strict_check: whether to raise an error when the meta information is insufficient.
            channel_dim: This argument can be used to specify the original channel dimension
                    (integer) of the input array.
                It overrides the `original_channel_dim` from provided MetaTensor input.
                If the input array doesn't have a channel dim, this value should be
                    ``'no_channel'``.
                If this is set to `None`, this class relies on `img` or `meta_dict` to provide
                    the channel dimension.
        """
        super().__init__()

        self._ensure_channel_first = monai_utility_transforms.EnsureChannelFirst(
            strict_check=strict_check,
            channel_dim=channel_dim,
        )

    @functools.singledispatchmethod
    @override
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return inpt

    @_transform.register(tv_tensors.Image)
    @_transform.register(eva_tv_tensors.Volume)
    @_transform.register(tv_tensors.Mask)
    def _(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt_channel_first = self._ensure_channel_first(inpt)
        return tv_tensors.wrap(inpt_channel_first, like=inpt)
