"""Unit tests for the SegResNet backbone."""

import pytest
import torch

from eva.vision.models.networks.backbones.radiology import segresnet


@pytest.mark.parametrize(
    "in_channels, spatial_dims, out_indices, expected_shapes",
    [
        (
            1,
            3,
            4,
            [
                torch.Size([4, 16, 96, 96, 96]),
                torch.Size([4, 32, 48, 48, 48]),
                torch.Size([4, 64, 24, 24, 24]),
                torch.Size([4, 128, 12, 12, 12]),
            ],
        ),
        (
            1,
            2,
            2,
            [
                torch.Size([4, 64, 24, 24]),
                torch.Size([4, 128, 12, 12]),
            ],
        ),
    ],
)
def test_segresnet_forward_pass(
    in_channels: int, spatial_dims: int, out_indices: int, expected_shapes: torch.Size
) -> None:
    """Tests if the forward pass works with common inputs and outputs expected tensor shapes."""
    spatial_dim_size = (96, 96, 96) if spatial_dims == 3 else (96, 96)
    batch = torch.randn(4, in_channels, *spatial_dim_size)

    model = segresnet.SegResNetEncoder(
        in_channels=in_channels,
        init_filters=16,
        spatial_dims=spatial_dims,
        out_indices=out_indices,
    )
    features = model(batch)

    assert len(features) == len(expected_shapes)
    for feature, expected_shape in zip(features, expected_shapes, strict=False):
        assert feature.shape == expected_shape
