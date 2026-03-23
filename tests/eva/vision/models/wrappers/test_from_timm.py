"""TimmModel tests."""

from typing import Any, Dict, Tuple

import pytest
import torch

from eva.vision.models import wrappers


@pytest.mark.parametrize(
    "model_name, out_indices, model_kwargs, input_tensor, expected_len, expected_shape",
    [
        (
            "vit_small_patch16_224",
            1,
            None,
            torch.Tensor(2, 3, 224, 224),
            1,
            torch.Size([2, 384, 14, 14]),
        ),
        (
            "vit_small_patch16_224",
            3,
            None,
            torch.Tensor(2, 3, 224, 224),
            3,
            torch.Size([2, 384, 14, 14]),
        ),
        (
            "vit_small_patch16_224",
            3,
            {"dynamic_img_size": True},
            torch.Tensor(2, 3, 512, 512),
            3,
            torch.Size([2, 384, 32, 32]),
        ),
    ],
)
def test_timm_model(
    timm_model: wrappers.TimmModel,
    input_tensor: torch.Tensor,
    expected_len: int,
    expected_shape: torch.Size,
) -> None:
    """Tests the TimmModel wrapper."""
    outputs = timm_model(input_tensor)
    assert isinstance(outputs, list)
    assert len(outputs) == expected_len
    # individual
    assert isinstance(outputs[0], torch.Tensor)
    assert outputs[0].shape == expected_shape


@pytest.fixture(scope="function")
def timm_model(
    model_name: str,
    out_indices: int | Tuple[int, ...] | None,
    model_kwargs: Dict[str, Any] | None,
) -> wrappers.TimmModel:
    """TimmModel fixture."""
    return wrappers.TimmModel(
        model_name=model_name,
        out_indices=out_indices,
        model_kwargs=model_kwargs,
    )
