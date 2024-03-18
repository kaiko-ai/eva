"""Embeddings writer."""

import csv
import io
import os
from typing import Any, Dict, Sequence

import lightning.pytorch as pl
import torch
from lightning.pytorch import callbacks
from loguru import logger
from torch import multiprocessing, nn
from typing_extensions import override

from eva.core.callbacks.writers.typings import QUEUE_ITEM
from eva.core.models.modules.typings import INPUT_BATCH
from eva.core.utils import multiprocessing as eva_multiprocessing


class EmbeddingsWriter(callbacks.BasePredictionWriter):
    """Callback for writing generated embeddings to disk."""

    def __init__(
        self,
        output_dir: str,
        backbone: nn.Module | None = None,
        dataloader_idx_map: Dict[int, str] | None = None,
        group_key: str | None = None,
        overwrite: bool = True,
    ) -> None:
        """Initializes a new EmbeddingsWriter instance.

        This callback writes the embedding files in a separate process to avoid blocking the
        main process where the model forward pass is executed.

        Args:
            output_dir: The directory where the embeddings will be saved.
            backbone: A model to be used as feature extractor. If `None`,
                it will be expected that the input batch returns the features directly.
            dataloader_idx_map: A dictionary mapping dataloader indices to their respective
                names (e.g. train, val, test).
            group_key: The metadata key to group the embeddings by. If specified, the
                embedding files will be saved in subdirectories named after the group_key.
                If specified, the key must be present in the metadata of the input batch.
            overwrite: Whether to overwrite the output directory. Defaults to True.
        """
        super().__init__(write_interval="batch")

        self._output_dir = output_dir
        self._backbone = backbone
        self._dataloader_idx_map = dataloader_idx_map or {}
        self._group_key = group_key
        self._overwrite = overwrite

        self._write_queue: multiprocessing.Queue
        self._write_process: eva_multiprocessing.Process

    @override
    def on_predict_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        os.makedirs(self._output_dir, exist_ok=self._overwrite)
        self._initialize_write_process()
        self._write_process.start()

        if self._backbone is not None:
            self._backbone = self._backbone.to(pl_module.device)
            self._backbone.eval()

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

        embeddings = self._get_embeddings(prediction)
        for local_idx, global_idx in enumerate(batch_indices[: len(embeddings)]):
            input_name, save_name = self._construct_save_name(
                dataset.filename(global_idx), metadata, local_idx
            )
            embeddings_buffer, target_buffer = io.BytesIO(), io.BytesIO()
            torch.save(embeddings[local_idx].clone(), embeddings_buffer)
            torch.save(targets[local_idx], target_buffer)  # type: ignore
            item = QUEUE_ITEM(embeddings_buffer, target_buffer, input_name, save_name, split)
            self._write_queue.put(item)

        self._write_process.check_exceptions()

    @override
    def on_predict_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._write_queue.put(None)
        self._write_process.join()
        logger.info(f"Predictions and manifest saved to {self._output_dir}")

    def _initialize_write_process(self) -> None:
        self._write_queue = multiprocessing.Queue()
        self._write_process = eva_multiprocessing.Process(
            target=_process_write_queue, args=(self._write_queue, self._output_dir, self._overwrite)
        )

    def _get_embeddings(self, prediction: torch.Tensor) -> torch.Tensor:
        """Returns the embeddings from predictions."""
        if self._backbone is None:
            return prediction

        with torch.no_grad():
            return self._backbone(prediction)

    def _construct_save_name(self, input_name, metadata, local_idx):
        group_name = metadata[self._group_key][local_idx] if self._group_key else None
        save_name = os.path.splitext(input_name)[0] + ".pt"
        if group_name:
            save_name = os.path.join(group_name, save_name)
        return input_name, save_name


def _process_write_queue(
    write_queue: multiprocessing.Queue, output_dir: str, overwrite: bool = False
) -> None:
    manifest_file, manifest_writer = _init_manifest(output_dir, overwrite)
    while True:
        item = write_queue.get()
        if item is None:
            break

        prediction_buffer, target_buffer, input_name, save_name, split = QUEUE_ITEM(*item)
        _save_prediction(prediction_buffer, save_name, output_dir)
        _update_manifest(target_buffer, input_name, save_name, split, manifest_writer)

    manifest_file.close()


def _save_prediction(prediction_buffer: io.BytesIO, save_name: str, output_dir: str) -> None:
    save_path = os.path.join(output_dir, save_name)
    prediction = torch.load(io.BytesIO(prediction_buffer.getbuffer()), map_location="cpu")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(prediction, save_path)


def _init_manifest(output_dir: str, overwrite: bool = False) -> tuple[io.TextIOWrapper, Any]:
    manifest_path = os.path.join(output_dir, "manifest.csv")
    if os.path.exists(manifest_path) and not overwrite:
        raise FileExistsError(
            f"Manifest file already exists at {manifest_path}. This likely means that the "
            "embeddings have been computed before. Consider using `eva fit` instead "
            "of `eva predict_fit` or `eva predict`."
        )
    manifest_file = open(manifest_path, "w", newline="")
    manifest_writer = csv.writer(manifest_file)
    manifest_writer.writerow(["origin", "embeddings", "target", "split"])
    return manifest_file, manifest_writer


def _update_manifest(
    target_buffer: io.BytesIO,
    input_name: str,
    save_name: str,
    split: str | None,
    manifest_writer,
) -> None:
    target = torch.load(io.BytesIO(target_buffer.getbuffer()), map_location="cpu")
    manifest_writer.writerow([input_name, save_name, target.item(), split])
