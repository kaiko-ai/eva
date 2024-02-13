"""Model inference module."""

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing_extensions import override

from eva.models.modules import module
from eva.models.modules.typings import INPUT_BATCH, MODEL_TYPE


class InferenceModule(module.ModelModule):
    """An lightweight model module to perform inference."""

    def __init__(self, model: MODEL_TYPE) -> None:
        """Initializes the module.

        Args:
            model: The network to be used for inference.
        """
        super().__init__(metrics=None)

        self.model = model

    @override
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.model(tensor)

    @override
    def predict_step(
        self,
        batch: INPUT_BATCH,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> STEP_OUTPUT:
        data, targets, metadata = INPUT_BATCH(*batch)
        predictions = self(data)

        return predictions
