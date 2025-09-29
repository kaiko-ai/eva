"""Base class for torchvision.v2 transforms."""

import abc
from typing import Any, Dict, List

from torchvision.transforms import v2


class TorchvisionTransformV2(v2.Transform, abc.ABC):
    """Wrapper for torchvision.v2.Transform.

    This class ensures compatibility both with >=0.21.0 and older versions,
    as torchvision 0.21.0 introduced a new transform API where they
    renamed the following methods:

    - `_get_params` -> `make_params`
    - `_transform` -> `transform`
    """

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        """Called internally before calling transform() on each input."""
        return {}

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return self.make_params(flat_inputs)

    @abc.abstractmethod
    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        """Applies the transformation to the input."""
        raise NotImplementedError

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self.transform(inpt, params)
