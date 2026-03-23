"""Common semantic segmentation decoders.

This module contains implementations of different types of decoder models
used in semantic segmentation. These decoders convert the high-level features
output by an encoder into pixel-wise predictions for segmentation tasks.
"""

from torch import nn

from eva.vision.models.networks.decoders.segmentation import decoder2d, linear


class ConvDecoder1x1(decoder2d.Decoder2D):
    """A convolutional decoder with a single 1x1 convolutional layer."""

    def __init__(self, in_features: int, num_classes: int) -> None:
        """Initializes the decoder.

        Args:
            in_features: The hidden dimension size of the embeddings.
            num_classes: Number of output classes as channels.
        """
        super().__init__(
            layers=nn.Conv2d(
                in_channels=in_features,
                out_channels=num_classes,
                kernel_size=(1, 1),
            ),
        )


class ConvDecoderMS(decoder2d.Decoder2D):
    """A multi-stage convolutional decoder with upsampling and convolutional layers.

    This decoder applies a series of upsampling and convolutional layers to transform
    the input features into output predictions with the desired spatial resolution.

    This decoder is based on the `+ms` segmentation decoder from DINOv2
    (https://arxiv.org/pdf/2304.07193)
    """

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
                nn.Upsample(scale_factor=2),
                nn.Conv2d(64, num_classes, kernel_size=(3, 3), padding=(1, 1)),
            ),
        )


class SingleLinearDecoder(linear.LinearDecoder):
    """A simple linear decoder with a single fully connected layer."""

    def __init__(self, in_features: int, num_classes: int) -> None:
        """Initializes the decoder.

        Args:
            in_features: The hidden dimension size of the embeddings.
            num_classes: Number of output classes as channels.
        """
        super().__init__(
            layers=nn.Linear(
                in_features=in_features,
                out_features=num_classes,
            ),
        )
