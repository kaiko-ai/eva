"""Test the ViT-Adapter encoder."""

from typing import Tuple

import pytest
import timm
import torch
from timm.models import vision_transformer
from torch import nn

from eva.vision.models.networks.adapters import vit_adapter

_NUMBER_LEARNABLE_PARAMETER_VIT_UNFROZEN = 8155608
"""Number of learnable parameter if the ViT backbone is not frozen."""

_NUMBER_LEARNABLE_PARAMETER_VIT_FROZEN = 2438192
"""Number of learnable parameter if the ViT backbone is frozen."""


_HIDDEN_STATE_SHAPE_EXPECTED = [
    (192, 56, 56),
    (192, 28, 28),
    (192, 14, 14),
]
"""Shapes expected for the output hidden states from the ViT-Adapter."""


@pytest.mark.parametrize(
    "freeze_vit, learnable_parameters_expected",
    [
        (False, _NUMBER_LEARNABLE_PARAMETER_VIT_UNFROZEN),
        (True, _NUMBER_LEARNABLE_PARAMETER_VIT_FROZEN),
    ],
)
def test_vit_adapter_expected_learnable_parameters(
    vit_backbone: vision_transformer.VisionTransformer,
    input_tensor: torch.Tensor,
    freeze_vit: bool,
    learnable_parameters_expected: int,
) -> None:
    """Tests if the number of learnable parameters for frozen & unfrozen backbone."""
    encoder = vit_adapter.ViTAdapter(
        vit_backbone=vit_backbone,
        freeze_vit=freeze_vit,
    )
    hidden_states = encoder(input_tensor)[0]
    assert _extract_number_learnable_parameters(encoder) == learnable_parameters_expected
    for hidden_state, shape_expected in zip(
        hidden_states, _HIDDEN_STATE_SHAPE_EXPECTED, strict=False
    ):
        assert _extract_hidden_state_shape(hidden_state) == shape_expected


def test_expected_output_values(vit_backbone: vision_transformer.VisionTransformer):
    """Tests if the output values are as expected."""
    input_tensor = torch.randn((1, 3, 224, 224), generator=torch.Generator().manual_seed(42))
    encoder = vit_adapter.ViTAdapter(
        vit_backbone=vit_backbone,
        freeze_vit=True,
    )
    output_tensors = encoder(input_tensor)

    # Use the tensor sum as proxy for the output tensor values
    pytest.approx(output_tensors[0].sum().item(), abs=1e-6) == -0.0009918212890625
    pytest.approx(output_tensors[1].sum().item(), abs=1e-6) == 0.000110626220703125
    pytest.approx(output_tensors[2].sum().item(), abs=1e-6) == -3.24249267578125e-05


def _extract_hidden_state_shape(hidden_state: torch.Tensor) -> Tuple[int]:
    """Extracts the shape of the hidden state."""
    return hidden_state.detach().numpy().shape


def _extract_number_learnable_parameters(model: nn.Module) -> int:
    """Extracts the number of learnable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


@pytest.fixture
def vit_backbone() -> vision_transformer.VisionTransformer:
    """Fixture for the ViT backbone."""
    return timm.create_model(model_name="vit_tiny_patch16_224", pretrained=False)


@pytest.fixture
def input_tensor() -> torch.Tensor:
    """Fixture for the input tensor."""
    return torch.randn((1, 3, 224, 224), generator=torch.Generator().manual_seed(42))
