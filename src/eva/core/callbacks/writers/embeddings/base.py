"""Embeddings writer base class."""

import abc
import io
import os
from typing import Any, Dict, List, Sequence

import lightning.pytorch as pl
import torch
from lightning.pytorch import callbacks
from loguru import logger
from torch import multiprocessing, nn
from typing_extensions import override

from eva.core import utils
from eva.core.callbacks.writers.embeddings.typings import QUEUE_ITEM
from eva.core.models.modules.typings import INPUT_BATCH
from eva.core.utils import multiprocessing as eva_multiprocessing


class EmbeddingsWriter(callbacks.BasePredictionWriter, abc.ABC):
    """Callback for writing generated embeddings to disk."""

    def __init__(
        self,
        output_dir: str,
        backbone: nn.Module | None = None,
        dataloader_idx_map: Dict[int, str] | None = None,
        metadata_keys: List[str] | None = None,
        overwrite: bool = False,
        save_every_n: int = 100,
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
            metadata_keys: An optional list of keys to extract from the batch metadata and store
                as additional columns in the manifest file.
            overwrite: Whether to overwrite if embeddings are already present in the specified
                output directory. If set to `False`, an error will be raised if embeddings are
                already present (recommended).
            save_every_n: Interval for number of iterations to save the embeddings to disk.
                During this interval, the embeddings are accumulated in memory.
        """
        super().__init__(write_interval="batch")

        self._output_dir = output_dir
        self._backbone = backbone
        self._dataloader_idx_map = dataloader_idx_map or {}
        self._overwrite = overwrite
        self._save_every_n = save_every_n
        self._metadata_keys = metadata_keys or []

        self._write_queue: multiprocessing.Queue
        self._write_process: eva_multiprocessing.Process

    @staticmethod
    @abc.abstractmethod
    def _process_write_queue(
        write_queue: multiprocessing.Queue,
        output_dir: str,
        metadata_keys: List[str],
        save_every_n: int,
        overwrite: bool = False,
    ) -> None:
        """This function receives and processes items added by the main process to the queue.

        Queue items contain the embedding tensors, targets and metadata which need to be
        saved to disk (.pt files and manifest).
        """

    @override
    def on_predict_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._check_if_exists()
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
        if not isinstance(targets, torch.Tensor):
            raise ValueError(f"Targets ({type(targets)}) should be `torch.Tensor`.")

        with torch.no_grad():
            embeddings = self._get_embeddings(prediction)

        for local_idx, global_idx in enumerate(batch_indices[: len(embeddings)]):
            data_name = dataset.filename(global_idx)
            save_name = os.path.splitext(data_name)[0] + ".pt"
            embeddings_buffer, target_buffer = _as_io_buffers(
                embeddings[local_idx], targets[local_idx]
            )
            item_metadata = self._get_item_metadata(metadata, local_idx)
            item = QUEUE_ITEM(
                prediction_buffer=embeddings_buffer,
                target_buffer=target_buffer,
                data_name=data_name,
                save_name=save_name,
                split=split,
                metadata=item_metadata,
            )
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
            target=self._process_write_queue,
            args=(
                self._write_queue,
                self._output_dir,
                self._metadata_keys,
                self._save_every_n,
                self._overwrite,
            ),
        )

    @abc.abstractmethod
    def _get_embeddings(self, tensor: torch.Tensor) -> torch.Tensor | List[List[torch.Tensor]]:
        """Returns the embeddings from predictions."""

    def _get_item_metadata(
        self, metadata: Dict[str, Any] | None, local_idx: int
    ) -> Dict[str, Any] | None:
        """Returns the metadata for the item at the given local index."""
        if not metadata:
            if self._metadata_keys:
                raise ValueError("Metadata keys are provided but the batch metadata is empty.")
            else:
                return None

        item_metadata = {}
        for key in self._metadata_keys:
            if key not in metadata:
                raise KeyError(f"Metadata key '{key}' not found in the batch metadata.")
            metadata_value = metadata[key][local_idx]
            try:
                item_metadata[key] = utils.to_cpu(metadata_value)
            except TypeError:
                item_metadata[key] = metadata_value

        return item_metadata

    def _check_if_exists(self) -> None:
        """Checks if the output directory already exists and if it should be overwritten."""
        os.makedirs(self._output_dir, exist_ok=True)
        if os.path.exists(os.path.join(self._output_dir, "manifest.csv")) and not self._overwrite:
            raise FileExistsError(
                f"The embeddings output directory already exists: {self._output_dir}. This "
                "either means that they have been computed before or that a wrong output "
                "directory is being used. Consider using `eva fit` instead, selecting a "
                "different output directory or setting overwrite=True."
            )
        os.makedirs(self._output_dir, exist_ok=True)


def _as_io_buffers(*items: torch.Tensor | List[torch.Tensor]) -> Sequence[io.BytesIO]:
    """Loads torch tensors as io buffers."""
    buffers = [io.BytesIO() for _ in range(len(items))]
    for tensor, buffer in zip(items, buffers, strict=False):
        torch.save(utils.clone(tensor), buffer)
    return buffers
