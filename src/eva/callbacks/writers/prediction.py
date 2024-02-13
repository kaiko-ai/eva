"""Batch prediction writer."""

import csv
from pathlib import Path
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

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self._manifest_file = open(Path(self.output_dir, "manifest.csv"), "w", newline="")
        self._manifest_writer = csv.writer(self._manifest_file)

        self._manifest_writer.writerow(["filename", "prediction"])

    @override
    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: Any,
        batch_indices: Sequence[int] | None,
        batch: INPUT_BATCH,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        dataset = trainer.predict_dataloaders[dataloader_idx].dataset  # type: ignore

        for local_idx, global_idx in enumerate(batch_indices):
            if local_idx >= len(prediction):
                break

            save_dir = (
                Path(self.output_dir, batch["metadata"][self._group_key][local_idx])  # type: ignore
                if self._group_key
                else self.output_dir
            )
            filename = dataset.filename(global_idx)
            save_name = Path(filename).with_suffix(".pt")
            save_path = Path(save_dir, save_name)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(prediction[local_idx], save_path)
            self._manifest_writer.writerow([filename, save_name])

    @override
    def on_predict_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._manifest_file.close()
        logger.info(f"Predictions saved to {self.output_dir}")
