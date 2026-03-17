"""Unit tests for the Swin UNETR encoder."""

import pytest
import torch

from eva.vision.models.networks.backbones.radiology import swin_unetr


@pytest.mark.parametrize(
    "in_channels, spatial_dims, out_indices, expected_shapes",
    [
        (
            1,
            3,
            6,
            [
                torch.Size([4, 3, 96, 96, 96]),
                torch.Size([4, 3, 48, 48, 48]),
                torch.Size([4, 6, 24, 24, 24]),
                torch.Size([4, 12, 12, 12, 12]),
                torch.Size([4, 24, 6, 6, 6]),
                torch.Size([4, 48, 3, 3, 3]),
            ],
        ),
        (
            1,
            2,
            4,
            [
                torch.Size([4, 6, 24, 24]),
                torch.Size([4, 12, 12, 12]),
                torch.Size([4, 24, 6, 6]),
                torch.Size([4, 48, 3, 3]),
            ],
        ),
    ],
)
def test_forward_pass(
    in_channels: int, spatial_dims: int, out_indices: int, expected_shapes: torch.Size
) -> None:
    """Tests if the forward pass works with common inputs and outputs expected tensor shapes."""
    spatial_dim_size = (96, 96, 96) if spatial_dims == 3 else (96, 96)
    batch = torch.randn(4, in_channels, *spatial_dim_size)

    model = swin_unetr.SwinUNETREncoder(
        in_channels=in_channels,
        feature_size=3,
        spatial_dims=spatial_dims,
        out_indices=out_indices,
    )
    features = model(batch)

    assert len(features) == len(expected_shapes)
    for feature, expected_shape in zip(features, expected_shapes, strict=False):
        assert feature.shape == expected_shape


@pytest.mark.parametrize(
    "embeddings_type, expected_dim",
    [
        ("head", 48),
        ("multiscale", 72),
    ],
)
def test_embeddings_type(embeddings_type: str, expected_dim: int) -> None:
    """Tests if different embedding types produce the correct flattened output shape."""
    in_channels = 1
    feature_size = 3
    batch_size = 2
    spatial_dims = 2

    model = swin_unetr.SwinUNETREncoder(
        in_channels=in_channels,
        feature_size=feature_size,
        spatial_dims=spatial_dims,
        embeddings_type=embeddings_type,
        out_indices=None,
    )

    batch = torch.randn(batch_size, in_channels, 96, 96)
    output = model(batch)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, expected_dim)
