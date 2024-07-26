"""Densely neural network related layers."""

import torch
from torch import nn
from typing_extensions import override


class UpsampleDouble(nn.Module):
    """A module for upsampling a given tensor by a factor of 2.

    This module uses a ConvTranspose2d layer followed by an ELU
    activation to perform the upsampling operation.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initializes the UpsampleDouble module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """
        super().__init__()

        self._layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.GroupNorm(1, out_channels),
            nn.ELU(inplace=True),
        )

    @override
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of the UpsampleDouble module.

        Args:
            tensor: Input tensor.

        Returns:
            The upsampled tensor.
        """
        return self._layers(tensor)


class CALayer(nn.Module):
    """Channel-Wise Gated Layer for applying channel attention.

    This module computes channel-wise attention and applies it to the input tensor.
    """

    def __init__(self, channel: int, reduction: int = 8) -> None:
        """Initializes the CALayer module.

        Args:
            channel: Number of input channels.
            reduction: Reduction ratio for the channel attention.
        """
        super().__init__()

        self._avg_pool = nn.AdaptiveAvgPool2d(1)
        self._ca_block = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(1, channel // reduction),
            nn.ELU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(1, channel),
            nn.Sigmoid(),
        )

    @override
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CALayer module.

        Args:
            tensor: Input tensor.

        Returns:
            The tensor after applying channel-wise attention.
        """
        outputs = self._avg_pool(tensor)
        outputs = self._ca_block(outputs)
        return tensor * outputs


class DenseNetBlock(nn.Module):
    """A block used in DenseNet architecture.

    This block consists of a bottleneck layer followed by a convolution layer.
    The output of the block is concatenated with the input tensor.
    """

    def __init__(self, inplanes: int, growth_rate: int) -> None:
        """Initializes the DenseNetBlock module.

        Args:
            inplanes: Number of input channels.
            growth_rate: Growth rate for the block.
        """
        super().__init__()
        self._dense_block = nn.Sequential(
            nn.Conv2d(inplanes, 4 * growth_rate, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(1, 4 * growth_rate),
            nn.ELU(inplace=True),
            nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, growth_rate),
        )

    @override
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of the DenseNetBlock module.

        Args:
            tensor: Input tensor.

        Returns:
            The output of the block and concat with the input.
        """
        outputs = self._dense_block(tensor)
        return torch.cat([tensor, outputs], dim=1)


class DenseNetLayer(nn.Module):
    """A layer consisting of multiple DenseNet blocks.

    This layer stacks several DenseNet blocks and applies a Channel-Wise Gated Layer at the end.
    """

    def __init__(self, inplanes: int, growth_rate: int, steps: int) -> None:
        """Initializes the DenseNetLayer module.

        Args:
            inplanes: Number of input channels.
            growth_rate: Growth rate for the blocks.
            steps: Number of DenseNet blocks to stack.
        """
        super().__init__()

        layers = []
        for _ in range(steps):
            nn_block = [
                DenseNetBlock(inplanes=inplanes, growth_rate=growth_rate),
                nn.ELU(inplace=True),
            ]
            layers.extend(nn_block)
            inplanes += growth_rate

        layers.append(CALayer(inplanes))
        self._layers = nn.Sequential(*layers)

    @override
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of the DenseNetLayer module.

        Args:
            tensor: Input tensor.

        Returns:
            The output tensor after passing through the stacked DenseNet blocks and the CALayer.
        """
        return self._layers(tensor)


class DenselyDecoder(nn.Module):
    """A decoder network consisting of multiple DenseNet layers and upsampling layers.

    This network progressively upsamples the input tensor while applying DenseNet blocks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        growth_rate: int = 16,
        steps: int = 3,
        scale_factor: int = 2,
    ) -> None:
        """Initializes the DenselyDecoder module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            growth_rate: Initial growth rate for the DenseNet blocks. Default is 16.
            steps: Number of DenseNet blocks per layer. Default is 3.
            scale_factor: Number of upsampling steps. Default is 2.
        """
        super().__init__()

        layers = []
        for _ in range(scale_factor):
            layers.append(DenseNetLayer(in_channels, growth_rate, steps))
            in_channels = in_channels + growth_rate * steps
            layers.append(UpsampleDouble(in_channels, in_channels // 2))
            in_channels = in_channels // 2
            growth_rate = growth_rate // 2

        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(1, out_channels),
            )
        )

        self._layers = nn.Sequential(*layers)

    @override
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of the DenselyDecoder module.

        Args:
            tensor: Input tensor.

        Returns:
            The output tensor after passing through the DenseNet layers and upsampling layers.
        """
        return self._layers(tensor)
