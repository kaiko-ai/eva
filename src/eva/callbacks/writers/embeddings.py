"""Embeddings writer."""

import csv
import os
from typing import Any, Sequence

import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning import callbacks
from pytorch_lightning.trainer.states import RunningStage
from typing_extensions import override

from eva.models.modules.typings import INPUT_BATCH


class EmbeddingsWriter(callbacks.BasePredictionWriter):
    """Callback for writing generated embeddings to disk."""

    def __init__(
        self, output_dir: str, dataloader_idx_map: dict | None, group_key: str | None = None
    ):
        """Initializes a new EmbeddingsWriter instance.

        Args:
            output_dir: The directory where the predictions will be saved.
            dataloader_idx_map: A dictionary mapping dataloader indices to their respective
                names (e.g. train, val, test).
            group_key: The metadata key to group the predictions by. If specified, the
                predictions will be saved in subdirectories named after the group key.
                If specified, the key must be present in the metadata of the input batch.
        """
        super().__init__(write_interval="batch")

        self.output_dir = output_dir

        self._group_key = group_key
        self._dataloader_idx_map = dataloader_idx_map if dataloader_idx_map else {}

        self._manifest_file = None
        self._manifest_writer = None

    @override
    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        super().setup(trainer, pl_module, stage)
        os.makedirs(self.output_dir, exist_ok=True)

        if trainer.state.stage == RunningStage.PREDICTING:
            self._init_manifest()

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
        _, targets, metadata = INPUT_BATCH(*batch)

        for local_idx, global_idx in enumerate(batch_indices[: len(prediction)]):
            input_name = dataset.filename(global_idx)
            group_name = metadata[self._group_key][local_idx] if self._group_key else None  # type: ignore
            save_name = self._save_prediction(prediction[local_idx], input_name, group_name)
            target = targets[local_idx].item() if isinstance(targets, torch.Tensor) else None
            split = self._dataloader_idx_map.get(dataloader_idx)
            self._update_manifest(input_name, save_name, target, split)

    @override
    def on_predict_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._manifest_file:
            self._manifest_file.close()
            logger.info(f"Predictions saved to {self.output_dir}")

    def _save_prediction(self, prediction: Any, input_filename: str, group_name: str | None) -> str:
        save_name = os.path.splitext(input_filename)[0] + ".pt"
        if group_name:
            save_name = os.path.join(group_name, save_name)
        save_path = os.path.join(self.output_dir, save_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(prediction, save_path)
        return save_name

    def _init_manifest(self):
        manifest_path = os.path.join(self.output_dir, "manifest.csv")
        if os.path.exists(manifest_path):
            raise FileExistsError(f"Manifest file already exists at {manifest_path}")

        self._manifest_file = open(manifest_path, "w", newline="")
        self._manifest_writer = csv.writer(self._manifest_file)
        self._manifest_writer.writerow(["filename", "embedding", "target", "split"])

    def _update_manifest(
        self, input_name: str, prediction_name: str, target: int | float | None, split: str | None
    ) -> None:
        if self._manifest_writer:
            self._manifest_writer.writerow([input_name, prediction_name, target, split])
