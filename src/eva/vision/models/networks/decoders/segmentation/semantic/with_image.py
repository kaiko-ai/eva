"""Convolutional semantic segmentation decoders that use input image & feature maps as input."""

from typing import List

import torch
from torch import nn
from torchvision.transforms.functional import rgb_to_grayscale
from typing_extensions import override

from eva.vision.models.networks.decoders.segmentation import decoder2d
from eva.vision.models.networks.decoders.segmentation.typings import DecoderInputs


class ConvDecoderWithImage(decoder2d.Decoder2D):
    """A convolutional that in addition to encoded features, also takes the input image as input.

    In a first stage, the input features are upsampled and passed through a convolutional layer,
    while in the second stage, the input image channels are concatenated with the upsampled features
    and passed through additional convolutional blocks in order to combine the image prior
    information with the encoded features. Lastly, a 1x1 conv operation reduces the number of
    channels to the number of classes.
    """

    _default_hidden_dims = [64, 32, 32]

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        greyscale: bool = False,
        hidden_dims: List[int] | None = None,
    ) -> None:
        """Initializes the decoder.

        Args:
            in_features: The hidden dimension size of the embeddings.
            num_classes: Number of output classes as channels.
            greyscale: Whether to convert input images to greyscale.
            hidden_dims: List of hidden dimensions for the convolutional layers.
        """
        hidden_dims = hidden_dims or self._default_hidden_dims
        if len(hidden_dims) != 3:
            raise ValueError("Hidden dims must have 3 elements.")

        super().__init__(
            layers=nn.Sequential(
                nn.Upsample(scale_factor=2),
                Conv2dBnReLU(in_features, hidden_dims[0]),
            )
        )
        self.greyscale = greyscale

        additional_hidden_dims = 1 if greyscale else 3
        self.image_block = nn.Sequential(
            Conv2dBnReLU(hidden_dims[0] + additional_hidden_dims, hidden_dims[1]),
            Conv2dBnReLU(hidden_dims[1], hidden_dims[2]),
        )

        self.classifier = nn.Conv2d(hidden_dims[2], num_classes, kernel_size=1)

    @override
    def forward(self, decoder_inputs: DecoderInputs) -> torch.Tensor:
        if decoder_inputs.images is None:
            raise ValueError("Input images are missing.")

        logits = super().forward(decoder_inputs)
        in_images = (
            rgb_to_grayscale(decoder_inputs.images) if self.greyscale else decoder_inputs.images
        )
        logits = torch.cat([logits, in_images], dim=1)
        logits = self.image_block(logits)

        return self.classifier(logits)


class Conv2dBnReLU(nn.Sequential):
    """A single convolutional layer with batch normalization and ReLU activation."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1
    ) -> None:
        """Initializes the layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolutional kernel.
            padding: Padding size for the convolutional layer.
        """
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
