"""TimmModel tests."""

from typing import Any, Dict

import pytest
import torch

from eva.vision.models import networks


@pytest.mark.parametrize(
    "model_name, model_arguments, input_tensor, expected_shape",
    [
        (
            "vit_small_patch16_224",
            {"num_classes": 0},
            torch.Tensor(2, 3, 224, 224),
            torch.Size([2, 384]),
        ),
        (
            "vit_small_patch16_224",
            {"num_classes": 0, "dynamic_img_size": True},
            torch.Tensor(2, 3, 512, 512),
            torch.Size([2, 384]),
        ),
    ],
)
def test_timm_model(
    timm_model: networks.TimmModel,
    input_tensor: torch.Tensor,
    expected_shape: torch.Size,
) -> None:
    """Tests the timm_model network."""
    outputs = timm_model(input_tensor)
    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == expected_shape


@pytest.fixture(scope="function")
def timm_model(
    model_name: str,
    model_arguments: Dict[str, Any] | None,
) -> networks.TimmModel:
    """TimmModel fixture."""
    return networks.TimmModel(
        model_name=model_name,
        model_arguments=model_arguments,
    )
