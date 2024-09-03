from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional

from eva.vision.models.networks.decoders import decoder


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0), nn.BatchNorm2d(out_c)
        )

    def forward(self, inputs):
        x1 = self.conv(inputs)
        x2 = self.shortcut(inputs)
        x = self.relu(x1 + x2)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r1 = ResidualBlock(in_c + out_c, out_c)

    def forward(self, x, s):
        x = self.up(x)
        x = torch.cat([x, s], axis=1)
        x = self.r1(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, scale):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)
        self.r1 = ResidualBlock(in_c, out_c)

    def forward(self, inputs):
        x = self.up(inputs)
        x = self.r1(x)
        return x


class PVTFormer(nn.Module):
    def __init__(self, in_features: Tuple[int, int, int], out_channels: int) -> None:
        super().__init__()

        """ Channel Reduction """
        self.c1 = nn.Sequential(
            nn.Conv2d(in_features[0], 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(in_features[1], 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(in_features[2], 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # self.c1 = Conv2D(64, 64, kernel_size=1, padding=0)
        # self.c2 = Conv2D(128, 64, kernel_size=1, padding=0)
        # self.c3 = Conv2D(320, 64, kernel_size=1, padding=0)

        self.d1 = DecoderBlock(64, 64)
        self.d2 = DecoderBlock(64, 64)
        self.d3 = UpBlock(64, 64, 4)

        self.u1 = UpBlock(64, 64, 4)
        self.u2 = UpBlock(64, 64, 8)
        self.u3 = UpBlock(64, 64, 16)

        self.r1 = ResidualBlock(64 * 4, 64)
        self.y = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, features):
        """Encoder"""
        # pvt1 = self.backbone(inputs)
        # e1 = pvt1[0]     ## [-1, 64, h/4, w/4]
        # e2 = pvt1[1]     ## [-1, 128, h/8, w/8]
        # e3 = pvt1[2]     ## [-1, 320, h/16, w/16]

        e1, e2, e3 = features

        # e1 = e1.permute(0, 3, 1, 2)
        # e2 = e2.permute(0, 3, 1, 2)
        # e3 = e3.permute(0, 3, 1, 2)

        c1 = self.c1(e1)
        c2 = self.c2(e2)
        c3 = self.c3(e3)

        d1 = self.d1(c3, c2)
        d2 = self.d2(d1, c1)
        d3 = self.d3(d2)

        u1 = self.u1(c1)
        u2 = self.u2(c2)
        u3 = self.u3(c3)

        x = torch.cat([d3, u1, u2, u3], axis=1)
        x = self.r1(x)
        y = self.y(x)
        return y


class PVTFormerDecoder(decoder.Decoder):
    """Convolutional segmentation decoder."""

    def __init__(self, in_features: Tuple[int, int, int], num_classes: int) -> None:
        """Initializes the convolutional based decoder head.

        Here the input nn layers will be directly applied to the
        features of shape (batch_size, hidden_size, n_patches_height,
        n_patches_width), where n_patches is image_size / patch_size.
        Note the n_patches is also known as grid_size.

        Args:
            layers: The convolutional layers to be used as the decoder head.
        """
        super().__init__()

        self._layers = PVTFormer(in_features=in_features, out_channels=num_classes)

    def _forward_head(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """Forward of the decoder head.

        Args:
            patch_embeddings: The patch embeddings tensor of shape
                (batch_size, hidden_size, n_patches_height, n_patches_width).

        Returns:
            The logits as a tensor (batch_size, n_classes, upscale_height, upscale_width).
        """
        return self._layers(patch_embeddings)

    def _cls_seg(
        self,
        logits: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Classify each pixel of the image.

        Args:
            logits: The decoder outputs of shape (batch_size, n_classes,
                height, width).
            image_size: The target image size (height, width).

        Returns:
            Tensor containing scores for all of the classes with shape
            (batch_size, n_classes, image_height, image_width).
        """
        return functional.interpolate(logits, image_size, mode="bilinear")

    def forward(
        self,
        features: List[torch.Tensor],
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Maps the patch embeddings to a segmentation mask of the image size.

        Args:
            features: List of multi-level image features of shape (batch_size,
                hidden_size, n_patches_height, n_patches_width).
            image_size: The target image size (height, width).

        Returns:
            Tensor containing scores for all of the classes with shape
            (batch_size, n_classes, image_height, image_width).
        """
        # patch_embeddings = self._forward_features(features)
        logits = self._forward_head(features)
        return self._cls_seg(logits, image_size)
