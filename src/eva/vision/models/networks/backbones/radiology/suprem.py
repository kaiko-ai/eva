"""Supervised PreTrained Models (SuPreM)."""

import collections

from eva.core.models.wrappers import _utils
from eva.vision.models.networks.backbones.radiology import segresnet, swin_unetr
from eva.vision.models.networks.backbones.registry import backbone_registry


@backbone_registry.register("radiology/suprem_swin_unetr")
class SuPreMSwinUNETR(swin_unetr.SwinUNETREncoder):  # type: ignore
    """SuPreM SwinUNETR Self-Supervised Encoder [0].

    - [0] Supervised Pre-Trained 3D Models for Medical Image Analysis
      https://github.com/MrGiovanni/SuPreM
    """

    _checkpoint: str = (
        "https://huggingface.co/MrGiovanni/SuPreM/resolve/main/supervised_suprem_swinunetr_2100.pth"
    )
    """Path to the model state dict."""

    _md5: str = "72688b8e5766e87f7c3712fa7555d1fd"
    """State dict MD5 validation code."""

    def __init__(self, out_indices: int | None = None) -> None:
        """Initializes the model.

        Args:
            out_indices: The number of feature maps from intermediate blocks
                to be returned. If set to 1, only the last feature map is returned.
        """
        super().__init__(
            in_channels=1,
            feature_size=48,
            spatial_dims=3,
            out_indices=out_indices,
            use_v2=False,
        )

        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """Loads the model checkpoint."""
        state_dict = _utils.load_state_dict_from_url(
            self._checkpoint,
            md5=self._md5,
        )

        encoder_state_dict = collections.OrderedDict(
            [
                (key.split("backbone.")[1], weights)
                for key, weights in state_dict["net"].items()
                if "encoder" in key or "swinViT" in key
            ]
        )
        self.load_state_dict(encoder_state_dict)


@backbone_registry.register("radiology/suprem_segresnet")
class SuPreMSegResNet(segresnet.SegResNetEncoder):  # type: ignore
    """SuPreM SegResNet Self-Supervised Encoder [0].

    - [0] Supervised Pre-Trained 3D Models for Medical Image Analysis
      https://github.com/MrGiovanni/SuPreM
    """

    _checkpoint: str = (
        "https://huggingface.co/MrGiovanni/SuPreM/resolve/main/supervised_suprem_segresnet_2100.pth"
    )
    """Path to the model state dict."""

    _md5: str = "fa044baf9a49832e7fe6ab55c9227e49"
    """State dict MD5 validation code."""

    def __init__(self, out_indices: int | None = None) -> None:
        """Initializes the model.

        Args:
            out_indices: The number of feature maps from intermediate blocks
                to be returned. If set to 1, only the last feature map is returned.
        """
        super().__init__(
            spatial_dims=3,
            init_filters=16,
            in_channels=1,
            out_indices=out_indices,
        )

        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """Loads the model checkpoint."""
        state_dict = _utils.load_state_dict_from_url(
            self._checkpoint,
            md5=self._md5,
        )

        encoder_state_dict = collections.OrderedDict(
            [
                (key.split("module.")[1], weights)
                for key, weights in state_dict["net"].items()
                if "convInit" in key or "down_layers" in key
            ]
        )
        self.load_state_dict(encoder_state_dict)
