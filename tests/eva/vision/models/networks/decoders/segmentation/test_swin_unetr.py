"""Unit test for the SwinUNETRDecoder."""

from typing import List

import pytest
import torch

from eva.vision.models.networks.backbones.radiology.swin_unetr import SwinUNETREncoder
from eva.vision.models.networks.decoders.segmentation.semantic import (
    SwinUNETRDecoder,
    SwinUNETRDecoderWithProjection,
)


@pytest.mark.parametrize("spatial_dims", [2, 3])
def test_forward_pass(spatial_dims: int):
    """Tests if the full forward pass using generated features by encoder."""
    in_channels, num_classes, feature_size, out_indices = 1, 7, 3, 6
    spatial_dim_size = (96, 96, 96) if spatial_dims == 3 else (96, 96)
    batch = torch.randn(4, in_channels, *spatial_dim_size)

    encoder = SwinUNETREncoder(
        in_channels=in_channels,
        feature_size=feature_size,
        spatial_dims=spatial_dims,
        out_indices=out_indices,
    )
    decoder = SwinUNETRDecoder(
        out_channels=num_classes,
        feature_size=feature_size,
        spatial_dims=spatial_dims,
    )
    features = encoder(batch)
    assert len(features) == out_indices

    output = decoder(features)
    assert output.shape == torch.Size([4, num_classes, *spatial_dim_size])


@pytest.mark.parametrize("spatial_dims", [2, 3])
@pytest.mark.parametrize("feature_size", [3, 6])
@pytest.mark.parametrize("out_channels", [1, 2, 5])
def test_swin_unetr_decoder_shapes(spatial_dims: int, feature_size: int, out_channels: int):
    """Tests SwinUNETRDecoder with different configurations."""
    batch_size = 2
    spatial_dim_size = (32, 32, 32) if spatial_dims == 3 else (32, 32)

    decoder = SwinUNETRDecoder(
        out_channels=out_channels,
        feature_size=feature_size,
        spatial_dims=spatial_dims,
    )

    features = _create_mock_features(batch_size, feature_size, spatial_dim_size)
    output = decoder(features)
    assert output.shape == torch.Size([batch_size, out_channels, *spatial_dim_size])


@pytest.mark.parametrize("spatial_dims", [2, 3])
@pytest.mark.parametrize("feature_size", [3, 6])
@pytest.mark.parametrize("out_channels", [1, 2])
def test_swin_unetr_decoder_with_projection_shapes(
    spatial_dims: int, feature_size: int, out_channels: int
):
    """Tests SwinUNETRDecoderWithProjection with different configurations."""
    batch_size = 2
    spatial_dim_size = (32, 32, 32) if spatial_dims == 3 else (32, 32)

    decoder = SwinUNETRDecoderWithProjection(
        out_channels=out_channels,
        feature_size=feature_size,
        spatial_dims=spatial_dims,
    )

    features = _create_mock_features(batch_size, feature_size, spatial_dim_size)
    output = decoder(features)
    assert output.shape == torch.Size([batch_size, out_channels, *spatial_dim_size])


@pytest.mark.parametrize("spatial_dims", [2, 3])
def test_swin_unetr_decoder_with_projection_custom_dims(spatial_dims: int):
    """Tests SwinUNETRDecoderWithProjection with custom projection dimensions."""
    batch_size = 2
    feature_size = 6
    out_channels = 2
    spatial_dim_size = (32, 32, 32) if spatial_dims == 3 else (32, 32)
    project_dims = [3, 4, 6, 8, 12, 16]

    decoder = SwinUNETRDecoderWithProjection(
        out_channels=out_channels,
        feature_size=feature_size,
        spatial_dims=spatial_dims,
        project_dims=project_dims,
    )

    features = _create_mock_features(batch_size, feature_size, spatial_dim_size)
    output = decoder(features)
    assert output.shape == torch.Size([batch_size, out_channels, *spatial_dim_size])


def test_swin_unetr_decoder_with_projection_invalid_project_dims():
    """Tests SwinUNETRDecoderWithProjection with invalid projection dimensions."""
    with pytest.raises(ValueError, match="project_dims must have exactly 6 dimensions"):
        SwinUNETRDecoderWithProjection(
            out_channels=2,
            feature_size=6,
            spatial_dims=3,
            project_dims=[3, 4, 6, 8],  # Only 4 dims instead of 6
        )


def test_swin_unetr_decoder_deterministic():
    """Tests that the decoder produces deterministic outputs."""
    torch.manual_seed(42)

    batch_size = 1
    feature_size = 3
    out_channels = 2
    spatial_dims = 3
    spatial_dim_size = (64, 64, 64)

    decoder = SwinUNETRDecoder(
        out_channels=out_channels,
        feature_size=feature_size,
        spatial_dims=spatial_dims,
    )

    features = _create_mock_features(batch_size, feature_size, spatial_dim_size)

    output1 = decoder(features)
    output2 = decoder(features)

    assert torch.allclose(output1, output2, atol=1e-6)


def test_swin_unetr_decoder_with_projection_no_projection():
    """Tests SwinUNETRDecoderWithProjection without projection dimensions."""
    batch_size = 1
    feature_size = 3
    out_channels = 2
    spatial_dims = 3
    spatial_dim_size = (64, 64, 64)

    decoder = SwinUNETRDecoderWithProjection(
        out_channels=out_channels,
        feature_size=feature_size,
        spatial_dims=spatial_dims,
        project_dims=None,  # No projection
    )

    features = _create_mock_features(batch_size, feature_size, spatial_dim_size)
    output = decoder(features)
    assert output.shape == torch.Size([batch_size, out_channels, *spatial_dim_size])


def test_swin_unetr_decoder_wrong_features_length():
    """Tests decoder behavior with wrong number of features."""
    decoder = SwinUNETRDecoder(
        out_channels=2,
        feature_size=3,
        spatial_dims=3,
    )

    # Provide only 4 features instead of 6
    features = [
        torch.randn(1, 3, 16, 16, 16),
        torch.randn(1, 3, 8, 8, 8),
        torch.randn(1, 6, 4, 4, 4),
        torch.randn(1, 12, 2, 2, 2),
    ]

    with pytest.raises((ValueError, RuntimeError, IndexError)):
        decoder(features)


def _create_mock_features(
    batch_size: int,
    feature_size: int,
    spatial_dim_size: tuple[int, ...],
) -> List[torch.Tensor]:
    """Create mock features matching expected SwinUNETREncoder output.

    Args:
        batch_size: Batch size for the features.
        feature_size: Base feature size.
        spatial_dim_size: Spatial dimensions of the input.

    Returns:
        List of 6 feature tensors: [enc0, enc1, enc2, enc3, hid3, dec4].
    """
    return [
        torch.randn(batch_size, feature_size, *spatial_dim_size),  # enc0
        torch.randn(batch_size, feature_size, *(s // 2 for s in spatial_dim_size)),  # enc1
        torch.randn(batch_size, feature_size * 2, *(s // 4 for s in spatial_dim_size)),  # enc2
        torch.randn(batch_size, feature_size * 4, *(s // 8 for s in spatial_dim_size)),  # enc3
        torch.randn(batch_size, feature_size * 8, *(s // 16 for s in spatial_dim_size)),  # hid3
        torch.randn(batch_size, feature_size * 16, *(s // 32 for s in spatial_dim_size)),  # dec4
    ]
