"""Wrapper class for ONNX models."""

from typing import Any, Callable, Literal

import onnxruntime as ort
import torch
from typing_extensions import override

from eva.core.models.wrappers import base


class ONNXModel(base.BaseModel[torch.Tensor, torch.Tensor]):
    """Wrapper class for loading ONNX models."""

    def __init__(
        self,
        path: str,
        device: Literal["cpu", "cuda"] | None = "cpu",
        transforms: Callable | None = None,
    ):
        """Initializes the model.

        Args:
            path: The path to the .onnx model file.
            device: The device to run the model on. This can be either "cpu" or "cuda".
            transforms: The transforms to apply to the output tensor produced by the model.
        """
        super().__init__(transforms=transforms)

        self._path = path
        self._device = device

        self.load_model()

    @override
    def load_model(self) -> Any:
        if self._device == "cuda" and not torch.cuda.is_available():
            raise ValueError("Device is set to 'cuda', but CUDA is not available.")
        provider = "CUDAExecutionProvider" if self._device == "cuda" else "CPUExecutionProvider"
        self._model = ort.InferenceSession(self._path, providers=[provider])  # type: ignore

    @override
    def model_forward(self, tensor: torch.Tensor) -> torch.Tensor:
        # TODO: Use IO binding to avoid copying the tensor to CPU.
        # https://onnxruntime.ai/docs/api/python/api_summary.html#data-on-device
        if not isinstance(self._model, ort.InferenceSession):
            raise ValueError("Model is not loaded.")
        inputs = {self._model.get_inputs()[0].name: tensor.detach().cpu().numpy()}
        outputs = self._model.run(None, inputs)[0]
        return torch.from_numpy(outputs).float().to(tensor.device)
