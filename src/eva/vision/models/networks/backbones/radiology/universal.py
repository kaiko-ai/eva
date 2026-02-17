"""Universal Model PreTrained Models."""

import collections

from eva.core.models.wrappers import _utils
from eva.vision.models.networks.backbones.radiology import swin_unetr
from eva.vision.models.networks.backbones.registry import backbone_registry


@backbone_registry.register("radiology/universal_swinunetr")
class UniversalModelSwinUNETR(swin_unetr.SwinUNETREncoder):  # type: ignore
    """CLIP-Driven Universal Model for Organ Segmentation and Tumor Detection [0].

    - [0] Clip-driven universal model for organ segmentation and tumor detection
      https://arxiv.org/abs/2301.00785
    - [1] CLIP-Driven Universal Model repository
      https://github.com/ljwztc/CLIP-Driven-Universal-Model
    """

    _checkpoint: str = (
        "https://huggingface.co/ljwztc/CLIP-Driven-Universal-Model/resolve/main/clip_driven_universal_swin_unetr.pth"
    )
    """Path to the model state dict."""

    _md5: str = "1ad7a68e904cee85141b05c6bf646ace"
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
                (key.split("module.")[1], weights)
                for key, weights in state_dict["net"].items()
                if "encoder" in key or "swinViT" in key
            ]
        )
        self.load_state_dict(encoder_state_dict)
