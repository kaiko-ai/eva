"""VoCo Self-Supervised Encoders."""

from typing_extensions import override

from eva.core.models.wrappers import _utils
from eva.vision.models.networks.backbones.radiology import swin_unetr
from eva.vision.models.networks.backbones.registry import backbone_registry


class _VoCo(swin_unetr.SwinUNETREncoder):  # type: ignore
    """Base class for the VoCo self-supervised encoders."""

    _checkpoint: str
    """Path to the model state dict."""

    _md5: str | None = None
    """State dict MD5 validation code."""

    def __init__(self, feature_size: int, out_indices: int | None = None) -> None:
        """Initializes the model.

        Args:
            feature_size: Size of the last feature map of SwinUNETR.
            out_indices: The number of feature maps from intermediate blocks
                to be returned. If set to 1, only the last feature map is returned.
        """
        super().__init__(
            in_channels=1,
            feature_size=feature_size,
            spatial_dims=3,
            out_indices=out_indices,
        )

        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """Loads the model checkpoint."""
        state_dict = _utils.load_state_dict_from_url(self._checkpoint, md5=self._md5)
        self.load_state_dict(state_dict)


@backbone_registry.register("radiology/voco_b")
class VoCoB(_VoCo):
    """VoCo Self-supervised pre-trained B model."""

    _checkpoint = "https://huggingface.co/Luffy503/VoCo/resolve/main/VoCo_B_SSL_head.pt"
    _md5 = "f80c4da2f81d700bdae3df188f2057eb"

    @override
    def __init__(self, out_indices: int | None = None) -> None:
        super().__init__(feature_size=48, out_indices=out_indices)


@backbone_registry.register("radiology/voco_l")
class VoCoL(_VoCo):
    """VoCo Self-supervised pre-trained L model."""

    _checkpoint = "https://huggingface.co/Luffy503/VoCo/resolve/main/VoCo_L_SSL_head.pt"
    _md5 = "795095d1d43ef3808ec4c41798310136"

    @override
    def __init__(self, out_indices: int | None = None) -> None:
        super().__init__(feature_size=96, out_indices=out_indices)


@backbone_registry.register("radiology/voco_h")
class VoCoH(_VoCo):
    """VoCo Self-supervised pre-trained H model."""

    _checkpoint = "https://huggingface.co/Luffy503/VoCo/resolve/main/VoCo_H_SSL_head.pt"
    _md5 = "76f95a474736b60bf5b8aad94643744d"

    @override
    def __init__(self, out_indices: int | None = None) -> None:
        super().__init__(feature_size=192, out_indices=out_indices)
