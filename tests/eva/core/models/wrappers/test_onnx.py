"""ONNXModel wrapper tests."""

from collections.abc import Iterator
from pathlib import Path
from typing import Tuple

import pytest
import torch
from lightning.pytorch.demos import boring_classes

from eva.core.models import wrappers


@pytest.mark.parametrize(
    "input_shape,expected_output_shape",
    [
        ((1, 32), (1, 2)),
        ((4, 32), (4, 2)),
    ],
)
def test_onnx_model(
    model_path: str, input_shape: Tuple[int, ...], expected_output_shape: Tuple[int, ...]
) -> None:
    """Tests the forward pass using the ONNXModel wrapper."""
    model = wrappers.ONNXModel(path=model_path)
    model.eval()

    input_tensor = torch.rand(1, 32)
    output_tensor = model(input_tensor)

    assert output_tensor.shape == (1, 2)


@pytest.fixture
def model_path(tmp_path: Path) -> Iterator[str]:
    """Fixture that creates a temporary .onnx model file."""
    model = boring_classes.BoringModel()
    input_tensor = torch.randn(1, 32)
    file_path = tmp_path / "model.onnx"
    dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    model.to_onnx(
        file_path, input_sample=input_tensor, export_params=True, dynamic_axes=dynamic_axes
    )

    yield file_path.as_posix()
    file_path.unlink(missing_ok=True)
