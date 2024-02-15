"""Batch prediction writer."""

import csv
import os
from typing import Any, Sequence

import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning import callbacks
from typing_extensions import override

from eva.models.modules.typings import INPUT_BATCH


class BatchPredictionWriter(callbacks.BasePredictionWriter):
    """Callback for writing predictions to disk."""

    def __init__(self, output_dir: str, group_key: str | None = None):
        """Initializes a new BatchPredictionWriter instance.

        Args:
            output_dir: The directory where the predictions will be saved.
            group_key: The metadata key to group the predictions by. If specified, the
                predictions will be saved in subdirectories named after the group key.
                If specified, the key must be present in the metadata of the input batch.
        """
        super().__init__(write_interval="batch")

        self.output_dir = output_dir

        self._group_key = group_key

        os.makedirs(self.output_dir, exist_ok=True)
        self._manifest_file = open(os.path.join(self.output_dir, "manifest.csv"), "w", newline="")
        self._manifest_writer = csv.writer(self._manifest_file)

        self._manifest_writer.writerow(["filename", "prediction"])

    @override
    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: Any,
        batch_indices: Sequence[int],
        batch: INPUT_BATCH,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        dataset = trainer.predict_dataloaders[dataloader_idx].dataset  # type: ignore

        for local_idx, global_idx in enumerate(batch_indices[: len(prediction)]):
            save_dir = self._get_save_dir(batch, local_idx)
            self._save_prediction(prediction[local_idx], dataset.filename(global_idx), save_dir)

    @override
    def on_predict_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._manifest_file.close()
        logger.info(f"Predictions saved to {self.output_dir}")

    def _get_save_dir(self, batch: INPUT_BATCH, idx: int) -> str:
        save_dir = (
            os.path.join(self.output_dir, batch["metadata"][self._group_key][idx])  # type: ignore
            if self._group_key
            else self.output_dir
        )
        return save_dir

    def _save_prediction(self, prediction: torch.Tensor, file_name: str, save_dir: str):
        save_name = os.path.splitext(file_name)[0] + ".pt"
        save_path = os.path.join(save_dir, save_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(prediction, save_path)
        self._manifest_writer.writerow([file_name, save_name])
