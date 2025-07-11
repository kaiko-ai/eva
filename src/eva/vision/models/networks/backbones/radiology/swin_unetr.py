"""Encoder based on Swin UNETR."""

from typing import List, Tuple

import torch
from monai.inferers.inferer import Inferer
from monai.networks.blocks import unetr_block
from monai.networks.nets import swin_unetr
from monai.utils import misc
from torch import nn

from eva.vision.models.networks.backbones.registry import backbone_registry


@backbone_registry.register("radiology/swin_unetr_encoder")
class SwinUNETREncoder(nn.Module):
    """Swin transformer encoder based on UNETR [0].

    - [0] UNETR: Transformers for 3D Medical Image Segmentation
      https://arxiv.org/pdf/2103.10504
    """

    def __init__(
        self,
        in_channels: int = 1,
        feature_size: int = 48,
        spatial_dims: int = 3,
        out_indices: int | None = None,
        inferer: Inferer | None = None,
        use_v2: bool = True,
    ) -> None:
        """Build the UNETR encoder.

        Args:
            in_channels: Number of input channels.
            feature_size: The dimension of network feature size.
            spatial_dims: Number of spatial dimensions.
            out_indices: Number of feature outputs. If None,
                the aggregated feature vector is returned.
            inferer: An optional MONAI `Inferer` for efficient
                inference during evaluation.
            use_v2: Whether to use SwinTransformerV2.
        """
        super().__init__()

        self._in_channels = in_channels
        self._feature_size = feature_size
        self._spatial_dims = spatial_dims
        self._out_indices = out_indices
        self._inferer = inferer
        self._use_v2 = use_v2

        self._window_size = misc.ensure_tuple_rep(7, spatial_dims)
        self._patch_size = misc.ensure_tuple_rep(2, spatial_dims)

        self.swinViT = swin_unetr.SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=self._window_size,
            patch_size=self._patch_size,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=torch.nn.LayerNorm,
            spatial_dims=spatial_dims,
            use_v2=use_v2,
        )
        self.encoder1 = unetr_block.UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.encoder2 = unetr_block.UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.encoder3 = unetr_block.UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.encoder4 = unetr_block.UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.encoder10 = unetr_block.UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self._pool_op = (
            nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
            if spatial_dims == 3
            else nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

    def _forward_features(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Extracts feature maps from the Swin Transformer and encoder blocks.

        Args:
            tensor: Input tensor of shape (B, C, T, H, W).

        Returns:
            List of feature maps from encoder stages.
        """
        hidden_states = self.swinViT(tensor)
        enc0 = self.encoder1(tensor)
        enc1 = self.encoder2(hidden_states[0])
        enc2 = self.encoder3(hidden_states[1])
        enc3 = self.encoder4(hidden_states[2])
        dec4 = self.encoder10(hidden_states[4])
        return [enc0, enc1, enc2, enc3, hidden_states[3], dec4]

    def forward_features(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Computes feature maps using either standard forward pass or inference mode.

        If in inference mode (`self.training` is False) and an inference method
        (`self._inferer`) is available,  the `_inferer` is used to extract features.
        Otherwise, `_forward_features` is called directly.

        Args:
            tensor: Input tensor of shape (B, C, T, H, W).

        Returns:
            List of feature maps from encoder stages.
        """
        if not self.training and self._inferer:
            return self._inferer(inputs=tensor, network=self._forward_features)

        return self._forward_features(tensor)

    def forward_encoders(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Aggregates encoder features into a single feature vector.

        Args:
            features: List of feature maps from encoder stages.

        Returns:
            Aggregated feature vector (B, C').
        """
        batch_size = features[0].shape[0]
        reduced_features = []
        for patch_features in features[:4] + features[5:]:
            hidden_features = self._pool_op(patch_features)
            hidden_features_reduced = hidden_features.view(batch_size, -1)
            reduced_features.append(hidden_features_reduced)
        return torch.cat(reduced_features, dim=1)

    def forward_head(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Casts last feature map into a single feature vector.

        Args:
            features: List of encoder feature maps.

        Returns:
            Aggregated feature vector (B, C').
        """
        last_feature_map = features[-1]
        pooled_features = self._pool_op(last_feature_map)
        return torch.flatten(pooled_features, 1)

    def forward_embeddings(self, tensor: torch.Tensor) -> torch.Tensor:
        """Computes the final aggregated feature vector.

        Args:
            tensor: Input tensor of shape (B, C, T, H, W).

        Returns:
            Aggregated feature vector of shape (B, C').
        """
        intermediates = self.forward_features(tensor)
        return self.forward_encoders(intermediates)

    def forward_intermediates(
        self, tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Computes encoder features and their embeddings.

        Args:
            tensor: Input tensor of shape (B, C, T, H, W).

        Returns:
            Aggregated feature vector and list of intermediate features.
        """
        features = self.forward_features(tensor)
        embeddings = self.forward_encoders(features)
        return embeddings, features

    def forward(self, tensor: torch.Tensor) -> torch.Tensor | List[torch.Tensor]:
        """Forward pass through the encoder.

        If `self._out_indices` is None, it returns the aggregated feature vector.
        Otherwise, it returns the intermediate feature maps up to the specified index.

        Args:
            tensor: Input tensor of shape (B, C, T, H, W).

        Returns:
            Aggregated feature vector or intermediate features.
        """
        if self._out_indices is None:
            return self.forward_embeddings(tensor)

        intermediates = self.forward_features(tensor)
        return intermediates[-1 * self._out_indices :]
