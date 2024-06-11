"""Embeddings writer."""

import csv
import io
import os
from typing import Any, Dict, List, Sequence, Tuple

import lightning.pytorch as pl
import torch
from lightning.pytorch import callbacks
from loguru import logger
from torch import multiprocessing, nn
from typing_extensions import override

from eva.core.callbacks.writers.embeddings.typings import ITEM_DICT_ENTRY, QUEUE_ITEM
from eva.core.models.modules.typings import INPUT_BATCH
from eva.core.utils import multiprocessing as eva_multiprocessing


class EmbeddingsWriter(callbacks.BasePredictionWriter):
    """Callback for writing generated embeddings to disk."""

    def __init__(
        self,
        output_dir: str,
        backbone: nn.Module | None = None,
        dataloader_idx_map: Dict[int, str] | None = None,
        metadata_keys: List[str] | None = None,
        overwrite: bool = True,
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
            overwrite: Whether to overwrite the output directory. Defaults to True.
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
            input_name = dataset.filename(global_idx)
            save_name = os.path.splitext(input_name)[0] + ".pt"
            embeddings_buffer, target_buffer = io.BytesIO(), io.BytesIO()
            torch.save(embeddings[local_idx].clone(), embeddings_buffer)
            torch.save(targets[local_idx], target_buffer)  # type: ignore
            item_metadata = self._get_item_metadata(metadata, local_idx)
            item = QUEUE_ITEM(
                embeddings_buffer, target_buffer, input_name, save_name, split, item_metadata
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
            target=_process_write_queue,
            args=(
                self._write_queue,
                self._output_dir,
                self._metadata_keys,
                self._save_every_n,
                self._overwrite,
            ),
        )

    def _get_embeddings(self, prediction: torch.Tensor) -> torch.Tensor:
        """Returns the embeddings from predictions."""
        if self._backbone is None:
            return prediction

        with torch.no_grad():
            return self._backbone(prediction)

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
            item_metadata[key] = metadata[key][local_idx]

        return item_metadata


def _process_write_queue(
    write_queue: multiprocessing.Queue,
    output_dir: str,
    metadata_keys: List[str],
    save_every_n: int,
    overwrite: bool = False,
) -> None:
    """This function receives and processes items added by the main process to the queue."""
    manifest_file, manifest_writer = _init_manifest(output_dir, metadata_keys, overwrite)

    name_to_items: Dict[str, ITEM_DICT_ENTRY] = {}

    counter = 0
    while True:
        item = write_queue.get()
        if item is None:
            break

        item = QUEUE_ITEM(*item)

        if item.save_name in name_to_items:
            name_to_items[item.save_name].items.append(item)
        else:
            name_to_items[item.save_name] = ITEM_DICT_ENTRY(items=[item], save_count=0)

        if counter > 0 and counter % save_every_n == 0:
            name_to_items = _save_items(name_to_items, metadata_keys, output_dir, manifest_writer)

        counter += 1

    if len(name_to_items) > 0:
        _save_items(name_to_items, metadata_keys, output_dir, manifest_writer)

    manifest_file.close()


def _save_items(
    name_to_items: Dict[str, ITEM_DICT_ENTRY],
    metadata_keys: List[str],
    output_dir: str,
    manifest_writer: Any,
) -> Dict[str, ITEM_DICT_ENTRY]:
    """Saves predictions to disk and updates the manifest file.

    If multiple items share the same filename, the predictions are concatenated and saved
    to the same file. Furthermore, the manifest file will only contain one entry for each
    filename, which is why this function checks if it's the first time saving to a file.

    Args:
        name_to_items: A dictionary mapping save names to the corresponding queue items
            holding the prediction tensors and the information for the manifest file.
        metadata_keys: A list of keys to extract from the batch metadata. These will be
            stored as additional columns in the manifest file.
        output_dir: The directory where the embedding tensors & manifest will be saved.
        manifest_writer: The CSV writer for the writing to the manifest file.
    """
    for save_name, entry in name_to_items.items():
        if len(entry.items) > 0:
            save_path = os.path.join(output_dir, save_name)
            is_first_save = entry.save_count == 0
            if is_first_save:
                _, target, input_name, _, split, metadata = QUEUE_ITEM(*entry.items[0])
                metadata = [metadata[key] for key in metadata_keys]  # type: ignore
                _update_manifest(target, input_name, save_name, split, metadata, manifest_writer)
            prediction_buffers = [item.prediction_buffer for item in entry.items]
            _save_predictions(prediction_buffers, save_path, is_first_save)
            name_to_items[save_name].save_count += 1
            name_to_items[save_name].items = []

    return name_to_items


def _save_predictions(
    prediction_buffers: List[io.BytesIO], save_path: str, is_first_save: bool
) -> None:
    """Saves the embedding tensors as list to .pt files.

    If it's not the first save to this save_path, the new predictions are concatenated
    with the existing ones and saved to the same file.

    Example Usecase: Save all patch embeddings corresponding to the same WSI to a single file.
    """
    predictions = [
        torch.load(io.BytesIO(buffer.getbuffer()), map_location="cpu")
        for buffer in prediction_buffers
    ]

    if not is_first_save:
        previous_predictions = torch.load(save_path, map_location="cpu")
        if not isinstance(previous_predictions, list):
            raise ValueError("Previous predictions should be a list of tensors.")
        predictions = predictions + previous_predictions

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(predictions, save_path)


def _init_manifest(
    output_dir: str, metadata_keys: List[str] | None, overwrite: bool = False
) -> Tuple[io.TextIOWrapper, Any]:
    manifest_path = os.path.join(output_dir, "manifest.csv")
    if os.path.exists(manifest_path) and not overwrite:
        raise FileExistsError(
            f"Manifest file already exists at {manifest_path}. This likely means that the "
            "embeddings have been computed before. Consider using `eva fit` instead "
            "of `eva predict_fit` or `eva predict`."
        )
    manifest_file = open(manifest_path, "w", newline="")
    manifest_writer = csv.writer(manifest_file)
    manifest_writer.writerow(["origin", "embeddings", "target", "split"] + (metadata_keys or []))
    return manifest_file, manifest_writer


def _update_manifest(
    target_buffer: io.BytesIO,
    input_name: str,
    save_name: str,
    split: str | None,
    metadata: List[str],
    manifest_writer,
) -> None:
    target = torch.load(io.BytesIO(target_buffer.getbuffer()), map_location="cpu")
    manifest_writer.writerow([input_name, save_name, target.item(), split] + metadata)
