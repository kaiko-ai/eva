"""ModelFromRegistry tests."""

from typing import Any, Dict

import pytest
import torch

from eva.vision.models import wrappers


@pytest.mark.parametrize(
    "model_name, model_kwargs, input_tensor, expected_len, expected_shape",
    [
        (
            "universal/vit_small_patch16_224_random",
            {"out_indices": 1},
            torch.Tensor(2, 3, 224, 224),
            1,
            torch.Size([2, 384, 14, 14]),
        ),
        (
            "universal/vit_small_patch16_224_random",
            {"out_indices": 3},
            torch.Tensor(2, 3, 224, 224),
            3,
            torch.Size([2, 384, 14, 14]),
        ),
        (
            "universal/vit_small_patch16_224_random",
            {"dynamic_img_size": True, "out_indices": 3},
            torch.Tensor(2, 3, 512, 512),
            3,
            torch.Size([2, 384, 32, 32]),
        ),
    ],
)
def test_vision_backbone(
    backbone_model: wrappers.ModelFromRegistry,
    input_tensor: torch.Tensor,
    expected_len: int,
    expected_shape: torch.Size,
) -> None:
    """Tests the ModelFromRegistry wrapper."""
    outputs = backbone_model(input_tensor)
    assert isinstance(outputs, list)
    assert len(outputs) == expected_len
    # individual
    assert isinstance(outputs[0], torch.Tensor)
    assert outputs[0].shape == expected_shape


@pytest.fixture(scope="function")
def backbone_model(
    model_name: str,
    model_kwargs: Dict[str, Any] | None,
) -> wrappers.ModelFromRegistry:
    """ModelFromRegistry fixture."""
    return wrappers.ModelFromRegistry(
        model_name=model_name,
        model_kwargs=model_kwargs,
    )
