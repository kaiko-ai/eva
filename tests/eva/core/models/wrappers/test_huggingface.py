"""HuggingFaceModel wrapper tests."""

from typing import Callable, Tuple

import pytest
import torch
from transformers import modeling_outputs

from eva.core.models import transforms, wrappers


@pytest.mark.parametrize(
    "model_name_or_path, tensor_transforms, expected_output_shape",
    [
        ("hf-internal-testing/tiny-random-ViTModel", None, (16, 226, 32)),
        ("hf-internal-testing/tiny-random-ViTModel", transforms.ExtractCLSFeatures(), (16, 32)),
        (
            "hf-internal-testing/tiny-random-ViTModel",
            transforms.ExtractCLSFeatures(include_patch_tokens=True),
            (16, 64),
        ),
    ],
)
def test_huggingface_model(
    model_name_or_path: str,
    tensor_transforms: Callable | None,
    expected_output_shape: Tuple[int, ...],
) -> None:
    """Tests the forward pass using the HuggingFaceModel wrapper."""
    model = wrappers.HuggingFaceModel(model_name_or_path, tensor_transforms)
    input_tenor = torch.rand(16, 3, 30, 30)
    output_tensor = model(input_tenor)

    if isinstance(output_tensor, modeling_outputs.BaseModelOutputWithPooling):
        assert output_tensor.last_hidden_state.shape == expected_output_shape  # type: ignore
    else:
        assert output_tensor.shape == expected_output_shape
