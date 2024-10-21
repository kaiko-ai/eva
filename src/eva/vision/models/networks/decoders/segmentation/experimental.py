"""Convolutional based semantic segmentation decoder."""

from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional
from eva.vision.models.networks.decoders.segmentation import conv2d, linear
from torchvision.transforms import functional

from eva.vision.models.networks.decoders import decoder


class ConvDecoderMSWithReluAndBN(conv2d.ConvDecoder):
    def __init__(self, in_features: int, num_classes: int) -> None:
        """Initializes the decoder.

        Args:
            in_features: The hidden dimension size of the embeddings.
            num_classes: Number of output classes as channels.
        """
        super().__init__(
            layers=nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, 64, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(64, num_classes, kernel_size=(3, 3), padding=(1, 1)),
            ),
        )

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upsample: bool = True) -> None:
        super().__init__()
        
        layers = []
        if upsample:
            layers.append(nn.Upsample(scale_factor=2))
        
        layers.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])
        
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class ConvDecoderMSWithImagePrior(conv2d.ConvDecoder):
    def __init__(self, in_features: int, num_classes: int, use_input_image: bool=True, greyscale: bool=True, hidden_dim: int=64) -> None:
        """Initializes the decoder.

        Args:
            in_features: The hidden dimension size of the embeddings.
            num_classes: Number of output classes as channels.
        """
        super().__init__(
            layers = nn.Sequential(
                DecoderBlock(in_features, hidden_dim),
            )
        )
        self.greyscale = greyscale

        additional_hidden_dims = (1 if greyscale else 3) if use_input_image else 0
        self.image_block = nn.Sequential(
            DecoderBlock(hidden_dim + additional_hidden_dims, 32, upsample=False),
            DecoderBlock(32, 32, upsample=False),
        )

        self.cls = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(
        self,
        features: List[torch.Tensor],
        image_size: Tuple[int, int],
        in_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = super().forward(features, image_size)
        if in_image is not None:
            in_image = functional.rgb_to_grayscale(in_image) if self.greyscale else in_image
            logits = torch.cat([logits, in_image], dim=1)
        logits = self.image_block(logits)
        return self.cls(logits)
