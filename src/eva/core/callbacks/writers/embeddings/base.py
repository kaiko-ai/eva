"""Embeddings writer."""

import abc
import io
import os
from typing import Any, Dict, Sequence

import lightning.pytorch as pl
import torch
from lightning.pytorch import callbacks
from loguru import logger
from torch import multiprocessing, nn
from typing_extensions import override

from eva.core.callbacks.writers.embeddings.typings import QUEUE_ITEM
from eva.core.models.modules.typings import INPUT_BATCH
from eva.core.utils import multiprocessing as eva_multiprocessing


class EmbeddingsWriter(callbacks.BasePredictionWriter):
    """Base callback class for writing generated embeddings to disk."""

    def __init__(
        self,
        output_dir: str,
        backbone: nn.Module | None = None,
        dataloader_idx_map: Dict[int, str] | None = None,
        group_key: str | None = None,
        overwrite: bool = True,
    ) -> None:
        """Initializes a new EmbeddingsWriter instance.

        This callback writes the embedding files in a separate process
        to avoid blocking the main process where the model forward pass
        is executed.

        Args:
            output_dir: The directory where the embeddings will be saved.
            backbone: A model to be used as feature extractor. If `None`,
                it will be expected that the input batch returns the
                features directly.
            dataloader_idx_map: A dictionary mapping dataloader indices to
                their respective names (e.g. train, val, test).
            group_key: The metadata key to group the embeddings by. If specified,
                the embedding files will be saved in subdirectories named after
                the group_key. If specified, the key must be present in the metadata
                of the input batch.
            overwrite: Whether to overwrite the output directory.
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
        batch_split = self._dataloader_idx_map.get(dataloader_idx)
        embeddings = self._get_embeddings(prediction)
        _, targets, metadata = INPUT_BATCH(*batch)
        if not isinstance(targets, torch.Tensor):
            raise ValueError(f"Targets ({type(targets)}) should be `torch.Tensor`.")

        for local_idx, global_idx in enumerate(batch_indices[: len(embeddings)]):
            input_name, save_as = self._construct_save_name(
                dataset.filename(global_idx), metadata, local_idx
            )
            embeddings_buffer, target_buffer = _as_io_buffers(
                embeddings[local_idx], targets[local_idx]
            )
            self._write_queue.put(
                obj=QUEUE_ITEM(
                    embeddings_buffer,
                    target_buffer,
                    input_name=input_name,
                    save_name=save_as,
                    split=batch_split,
                )
            )

        self._write_process.check_exceptions()

    @override
    def on_predict_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._write_queue.put(None)
        self._write_process.join()
        logger.info(f"Embeddings are saved to {self._output_dir}")

    def _initialize_write_process(self) -> None:
        self._write_queue = multiprocessing.Queue()
        self._write_process = eva_multiprocessing.Process(
            target=self._process_write_queue,
            args=(self._write_queue, self._output_dir, self._overwrite),
        )

    @torch.no_grad()
    def _get_embeddings(self, tensor: torch.Tensor) -> torch.Tensor:
        """Returns the embeddings from predictions."""
        return self._backbone(tensor) if self._backbone else tensor

    def _construct_save_name(self, input_name, metadata, local_idx):
        group_name = metadata[self._group_key][local_idx] if self._group_key else None
        save_as = os.path.splitext(input_name)[0] + ".pt"
        if group_name:
            save_as = os.path.join(group_name, save_as)
        return input_name, save_as

    @staticmethod
    @abc.abstractmethod
    def _process_write_queue(
        write_queue: multiprocessing.Queue, output_dir: str, overwrite: bool = False
    ) -> None:
        """Gets the first item of the queue and saves it."""


def _as_io_buffers(*items: torch.Tensor) -> Sequence[io.BytesIO]:
    buffers = [io.BytesIO() for _ in range(len(items))]
    for tensor, buffer in zip(items, buffers, strict=False):
        torch.save(tensor.clone(), buffer)
    return buffers
