"""Helper wrapper class for timm models."""

from typing import Any, Dict

import timm

from eva.core.models.networks import wrappers


class TimmModel(wrappers.ModelFromFunction):
    """Wrapper class for timm models."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool = False,
        checkpoint_path: str = "",
        model_arguments: Dict[str, Any] | None = None,
    ) -> None:
        """Initializes and constructs the model.

        Args:
            model_name: Name of model to instantiate.
            pretrained: If set to `True`, load pretrained ImageNet-1k weights.
            checkpoint_path: Path of checkpoint to load.
            model_arguments: The extra callable function / class arguments.
        """
        super().__init__(
            path=timm.create_model,
            arguments={
                "model_name": model_name,
                "pretrained": pretrained,
                "checkpoint_path": checkpoint_path,
            }
            | (model_arguments or {}),
        )
