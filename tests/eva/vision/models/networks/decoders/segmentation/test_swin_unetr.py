"""Unit test for the SwinUNETRDecoder."""

import pytest
import torch

from eva.vision.models.networks.backbones.radiology.swin_unetr import SwinUNETREncoder
from eva.vision.models.networks.decoders.segmentation.semantic import SwinUNETRDecoder


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
