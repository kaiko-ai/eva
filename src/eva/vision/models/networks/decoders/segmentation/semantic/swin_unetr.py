"""Decoder based on Swin UNETR."""

from typing import List

import torch
from monai.networks.blocks import dynunet_block, unetr_block
from torch import nn


class SwinUNETRDecoder(nn.Module):
    """Swin transformer decoder based on UNETR [0].

    - [0] UNETR: Transformers for 3D Medical Image Segmentation
      https://arxiv.org/pdf/2103.10504
    """

    def __init__(
        self,
        out_channels: int,
        feature_size: int = 48,
        spatial_dims: int = 3,
    ) -> None:
        """Builds the decoder.

        Args:
            out_channels: Number of output channels.
            feature_size: Dimension of network feature size.
            spatial_dims: Number of spatial dimensions.
        """
        super().__init__()

        self.decoder5 = unetr_block.UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.decoder4 = unetr_block.UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.decoder3 = unetr_block.UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.decoder2 = unetr_block.UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.decoder1 = unetr_block.UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.out = dynunet_block.UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=out_channels,
        )

    def _forward_features(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Forward function for multi-level feature maps to a single one."""
        enc0, enc1, enc2, enc3, hid3, dec4 = features
        dec3 = self.decoder5(dec4, hid3)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        return self.out(out)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Maps the patch embeddings to a segmentation mask.

        Args:
            features: List of multi-level intermediate features from
                :class:`SwinUNETREncoder`.

        Returns:
            Tensor containing scores for all of the classes with shape
            (batch_size, n_classes, image_height, image_width).
        """
        return self._forward_features(features)


class SwinUNETRDecoderWithProjection(nn.Module):
    """Swin transformer decoder based on UNETR [0].

    This implementation adds additional projection layers to reduce
    the number of channels in the feature maps before applying the upscaling
    convolutional blocks. This reduces the number of trainable parameters
    significantly and is useful when scaling up the encoder architecture.

    - [0] UNETR: Transformers for 3D Medical Image Segmentation
      https://arxiv.org/pdf/2103.10504
    """

    def __init__(
        self,
        out_channels: int,
        feature_size: int = 48,
        spatial_dims: int = 3,
        project_dims: list[int] | None = None,
        checkpoint_path: str | None = None,
    ) -> None:
        """Builds the decoder.

        Args:
            out_channels: Number of output channels.
            feature_size: Dimension of network feature size.
            spatial_dims: Number of spatial dimensions.
            project_dims: List of 6 dimensions to project encoder features to.
                If None, uses default channel progression based on feature_size.
                This is not part of the original implementation, but helps
                to reduce the number of decoder parameters when scaling up
                the encoder architecture (feature_size).
            checkpoint_path: Path to the checkpoint file.
        """
        super().__init__()

        self._checkpoint_path = checkpoint_path
        self._project_dims = project_dims

        if project_dims is not None and len(project_dims) != 6:
            raise ValueError(
                f"project_dims must have exactly 6 dimensions, got {len(project_dims)}"
            )

        channel_dims = project_dims or [
            feature_size,
            feature_size,
            feature_size * 2,
            feature_size * 4,
            feature_size * 8,
            feature_size * 16,
        ]

        self.decoder5 = unetr_block.UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=channel_dims[5],
            out_channels=channel_dims[4],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.decoder4 = unetr_block.UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=channel_dims[4],
            out_channels=channel_dims[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.decoder3 = unetr_block.UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=channel_dims[3],
            out_channels=channel_dims[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.decoder2 = unetr_block.UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=channel_dims[2],
            out_channels=channel_dims[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.decoder1 = unetr_block.UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=channel_dims[1],
            out_channels=channel_dims[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.out = dynunet_block.UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=channel_dims[0],
            out_channels=out_channels,
        )

        if self._project_dims:
            conv_layer = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
            self.proj_enc0 = conv_layer(feature_size, self._project_dims[0], kernel_size=1)
            self.proj_enc1 = conv_layer(feature_size, self._project_dims[1], kernel_size=1)
            self.proj_enc2 = conv_layer(feature_size * 2, self._project_dims[2], kernel_size=1)
            self.proj_enc3 = conv_layer(feature_size * 4, self._project_dims[3], kernel_size=1)
            self.proj_hid3 = conv_layer(feature_size * 8, self._project_dims[4], kernel_size=1)
            self.proj_dec4 = conv_layer(feature_size * 16, self._project_dims[5], kernel_size=1)

    def _forward_features(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Forward function for multi-level feature maps to a single one."""
        enc0, enc1, enc2, enc3, hid3, dec4 = self._project_features(features)
        dec3 = self.decoder5(dec4, hid3)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        return self.out(out)

    def _project_features(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Projects features using 1x1 to reduce number of channels."""
        return (
            [
                self.proj_enc0(features[0]),
                self.proj_enc1(features[1]),
                self.proj_enc2(features[2]),
                self.proj_enc3(features[3]),
                self.proj_hid3(features[4]),
                self.proj_dec4(features[5]),
            ]
            if self._project_dims
            else features
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Maps the patch embeddings to a segmentation mask.

        Args:
            features: List of multi-level intermediate features from
                :class:`SwinUNETREncoder`.

        Returns:
            Tensor containing scores for all of the classes with shape
            (batch_size, n_classes, image_height, image_width).
        """
        return self._forward_features(features)
