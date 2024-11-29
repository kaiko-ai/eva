"""TorchHubModel tests."""

from typing import Any, Dict, Tuple

import pytest
import torch

from eva.core.models import wrappers


@pytest.mark.parametrize(
    "model_name, repo_or_dir, out_indices, model_kwargs, "
    "input_tensor, expected_len, expected_shape",
    [
        (
            "dinov2_vits14",
            "facebookresearch/dinov2:main",
            None,
            None,
            torch.Tensor(2, 3, 224, 224),
            None,
            torch.Size([2, 384]),
        ),
        (
            "dinov2_vits14",
            "facebookresearch/dinov2:main",
            1,
            None,
            torch.Tensor(2, 3, 224, 224),
            1,
            torch.Size([2, 384, 16, 16]),
        ),
        (
            "dinov2_vits14",
            "facebookresearch/dinov2:main",
            3,
            None,
            torch.Tensor(2, 3, 224, 224),
            3,
            torch.Size([2, 384, 16, 16]),
        ),
    ],
)
def test_torchhub_model(
    torchhub_model: wrappers.TorchHubModel,
    input_tensor: torch.Tensor,
    expected_len: int | None,
    expected_shape: torch.Size,
) -> None:
    """Tests the torch.hub model wrapper."""
    outputs = torchhub_model(input_tensor)
    if torchhub_model._out_indices is not None:
        assert isinstance(outputs, list) or isinstance(outputs, tuple)
        assert len(outputs) == expected_len
        assert isinstance(outputs[0], torch.Tensor)
        assert outputs[0].shape == expected_shape
    else:
        assert isinstance(outputs, torch.Tensor)
        assert outputs.shape == expected_shape


@pytest.fixture(scope="function")
def torchhub_model(
    model_name: str,
    repo_or_dir: str,
    out_indices: int | Tuple[int, ...] | None,
    model_kwargs: Dict[str, Any] | None,
) -> wrappers.TorchHubModel:
    """TorchHubModel fixture."""
    return wrappers.TorchHubModel(
        model_name=model_name,
        repo_or_dir=repo_or_dir,
        out_indices=out_indices,
        model_kwargs=model_kwargs,
        pretrained=False,
    )
