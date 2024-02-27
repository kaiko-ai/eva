"""Embeddings writer."""

import csv
import io
import os
from typing import Any, Dict, Sequence

import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning import callbacks
from pytorch_lightning.trainer.states import RunningStage
from torch import multiprocessing
from typing_extensions import override

from eva.callbacks.writers.typings import QUEUE_ITEM
from eva.models.modules.typings import INPUT_BATCH
from eva.utils import multiprocessing as eva_multiprocessing


class EmbeddingsWriter(callbacks.BasePredictionWriter):
    """Callback for writing generated embeddings to disk."""

    def __init__(
        self,
        output_dir: str,
        dataloader_idx_map: Dict[int, str] | None,
        group_key: str | None = None,
    ):
        """Initializes a new EmbeddingsWriter instance.

        This callback writes the embedding files in a seperate process to avoid blocking the
        main process where the model forward pass is executed.

        Args:
            output_dir: The directory where the embeddings will be saved.
            dataloader_idx_map: A dictionary mapping dataloader indices to their respective
                names (e.g. train, val, test).
            group_key: The metadata key to group the embeddings by. If specified, the
                embedding files will be saved in subdirectories named after the group_key.
                If specified, the key must be present in the metadata of the input batch.
        """
        super().__init__(write_interval="batch")
        self.output_dir = output_dir
        self._group_key = group_key
        self._dataloader_idx_map = dataloader_idx_map or {}

        self._write_queue: multiprocessing.Queue
        self._write_process: eva_multiprocessing.Process

    def _initialize_write_process(self):
        self._write_queue = multiprocessing.Queue()
        self._write_process = eva_multiprocessing.Process(
            target=self._process_write_queue, args=(self._write_queue, self.output_dir)
        )

    @override
    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        super().setup(trainer, pl_module, stage)

        if trainer.state.stage == RunningStage.PREDICTING:
            os.makedirs(self.output_dir, exist_ok=True)
            self._initialize_write_process()
            self._write_process.start()

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
        split = self._dataloader_idx_map.get(dataloader_idx)

        for local_idx, global_idx in enumerate(batch_indices[: len(prediction)]):
            input_name, save_name = self._construct_save_name(
                dataset.filename(global_idx), metadata, local_idx
            )

            prediction_buffer, target_buffer = io.BytesIO(), io.BytesIO()
            torch.save(prediction[local_idx].clone(), prediction_buffer)
            torch.save(targets[local_idx], target_buffer)  # type: ignore
            item = QUEUE_ITEM(prediction_buffer, target_buffer, input_name, save_name, split)
            self._write_queue.put(item)

        self._write_process.check_exceptions()

    @override
    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._write_queue.put(None)  # Signal to the writing process to terminate
        self._write_process.join()
        logger.info(f"Predictions and manifest saved to {self.output_dir}")
        self._write_process = None
        self._write_queue = None

    def _construct_save_name(self, input_name, metadata, local_idx):
        group_name = metadata[self._group_key][local_idx] if self._group_key else None
        save_name = os.path.splitext(input_name)[0] + ".pt"
        if group_name:
            save_name = os.path.join(group_name, save_name)
        return input_name, save_name

    @staticmethod
    def _process_write_queue(write_queue: multiprocessing.Queue, output_dir: str):
        manifest_file, manifest_writer = EmbeddingsWriter._init_manifest(output_dir)

        while True:
            item = write_queue.get()
            if item is None:
                break
            prediction_buffer, target_buffer, input_name, save_name, split = QUEUE_ITEM(*item)
            EmbeddingsWriter._save_prediction(prediction_buffer, save_name, output_dir)
            EmbeddingsWriter._update_manifest(
                target_buffer, input_name, save_name, split, manifest_writer
            )

        manifest_file.close()

    @staticmethod
    def _save_prediction(prediction_buffer: io.BytesIO, save_name: str, output_dir: str):
        save_path = os.path.join(output_dir, save_name)
        prediction = torch.load(io.BytesIO(prediction_buffer.getbuffer()), map_location="cpu")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(prediction, save_path)

    @staticmethod
    def _init_manifest(output_dir: str) -> tuple[io.TextIOWrapper, Any]:
        manifest_path = os.path.join(output_dir, "manifest.csv")
        if os.path.exists(manifest_path):
            raise FileExistsError(
                f"Manifest file already exists at {manifest_path}. This likely means that the "
                "embeddings have been computed before. Consider using `eva fit` instead "
                "of `eva predict_fit` or `eva predict`."
            )
        manifest_file = open(manifest_path, "w", newline="")
        manifest_writer = csv.writer(manifest_file)
        manifest_writer.writerow(["filename", "embedding", "target", "split"])

        return manifest_file, manifest_writer

    @staticmethod
    def _update_manifest(
        target_buffer: io.BytesIO,
        input_name: str,
        save_name: str,
        split: str | None,
        manifest_writer,
    ):
        target = torch.load(io.BytesIO(target_buffer.getbuffer()), map_location="cpu")
        manifest_writer.writerow([input_name, save_name, target.item(), split])
