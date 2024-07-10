import torch
from torch import nn
from typing_extensions import override


class UpsampleDouble(nn.Module):
    """A module for upsampling a given tensor by a factor of 2."""

    def __init__(self, in_channels: int, out_channels: int):
        """Builds the nn module.

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
            nn.ELU(inplace=True),
        )

    @override
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._layers(tensor)


class CALayer(nn.Module):
    """ChannelWise Gated Layer."""

    def __init__(self, channel: int, reduction: int = 8) -> None:
        """Builds the nn module.

        Args:
            channel: Number of input channels.
            reduction: Reduction ratio for the channel attention.
        """
        super().__init__()

        self._avg_pool = nn.AdaptiveAvgPool2d(1)
        self._ca_block = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, padding=0),
            nn.ELU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    @override
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        outputs = self._avg_pool(tensor)
        outputs = self._ca_block(outputs)
        return tensor * outputs


class DenseNetBlock(nn.Module):
    def __init__(self, inplanes: int, growth_rate: int):
        super().__init__()
        self._dense_block = nn.Sequential(
            nn.Conv2d(inplanes, 4 * growth_rate, kernel_size=1, stride=1, padding=0),
            nn.ELU(inplace=True),
            nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1),
        )

    @override
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        outputs = self._dense_block(tensor)
        return torch.cat([tensor, outputs], dim=1)


class DenseNetLayer(nn.Module):
    def __init__(self, inplanes: int, growth_rate: int, steps: int) -> None:
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
        return self._layers(tensor)


class DenselyNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, growth_rate, steps, blocks, act=None, drop_prob=0.0):
        super().__init__()
        # downscale block
        net = []
        for i in range(blocks):
            net.append(DenseNetLayer(in_channels, growth_rate, steps))
            in_channels = in_channels + growth_rate * steps

        # output layer
        net.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))

        self._layers = nn.Sequential(*net)            

    def forward(self, tensor):
        return self._layers(tensor)


class DenselyDecoder(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, growth_rate=16, steps=3, scale_factor=2
    ):
        super().__init__()

        layers = []
        for _ in range(scale_factor):
            layers.append(DenseNetLayer(in_channels, growth_rate, steps))
            in_channels = in_channels + growth_rate * steps
            layers.append(UpsampleDouble(in_channels, in_channels // 2))
            in_channels = in_channels // 2
            growth_rate = growth_rate // 2

        # output block
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))

        self._layers = nn.Sequential(*layers)

    @override
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._layers(tensor)
