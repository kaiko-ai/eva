"""Defines the AsDiscrete transformation."""

import torch
from monai.networks.utils import one_hot


class AsDiscrete:
    def __init__(
        self,
        argmax: bool = False,
        to_onehot: int | bool | None = None,
        threshold: float | None = None,
    ) -> None:
        """Convert the input tensor/array into discrete values.

        Args:
            argmax: Whether to execute argmax function on input data before transform.
            to_onehot: if not None, convert input data into the one-hot format with
                specified number of classes. If bool, it will try to infer the number
                of classes.
            threshold: If not None, threshold the float values to int number 0 or 1
                with specified threshold.
        """
        super().__init__()

        self._argmax = argmax
        self._to_onehot = to_onehot
        self._threshold = threshold

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._argmax:
            tensor = torch.argmax(tensor, dim=1, keepdim=True)

        if self._to_onehot is not None:
            tensor = one_hot(tensor, num_classes=self._to_onehot, dim=1, dtype=torch.long)

        if self._threshold is not None:
            tensor = tensor >= self._threshold

        return tensor
