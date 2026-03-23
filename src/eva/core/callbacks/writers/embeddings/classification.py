"""Embeddings writer for classification."""

import io
import os
from typing import Dict, List

import torch
from torch import multiprocessing
from typing_extensions import override

from eva.core.callbacks.writers.embeddings import base
from eva.core.callbacks.writers.embeddings._manifest import ManifestManager
from eva.core.callbacks.writers.embeddings.typings import ITEM_DICT_ENTRY, QUEUE_ITEM


class ClassificationEmbeddingsWriter(base.EmbeddingsWriter):
    """Callback for writing generated embeddings to disk for classification tasks."""

    @staticmethod
    @override
    def _process_write_queue(
        write_queue: multiprocessing.Queue,
        output_dir: str,
        metadata_keys: List[str],
        save_every_n: int,
        overwrite: bool = False,
    ) -> None:
        """Processes the write queue and saves the predictions to disk.

        Note that in Multi Instance Learning (MIL) scenarios, we can have multiple
        embeddings per input data point. In that case, this function will save all
        embeddings that correspond to the same data point as a list of tensors to
        the same .pt file.
        """
        manifest_manager = ManifestManager(output_dir, metadata_keys, overwrite)
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
                name_to_items = _save_items(name_to_items, output_dir, manifest_manager)
            counter += 1

        if len(name_to_items) > 0:
            _save_items(name_to_items, output_dir, manifest_manager)

        manifest_manager.close()

    @override
    def _get_embeddings(self, tensor: torch.Tensor) -> torch.Tensor:
        """Returns the embeddings from predictions."""
        return self._backbone(tensor) if self._backbone else tensor


def _save_items(
    name_to_items: Dict[str, ITEM_DICT_ENTRY],
    output_dir: str,
    manifest_manager: ManifestManager,
) -> Dict[str, ITEM_DICT_ENTRY]:
    """Saves predictions to disk and updates the manifest file.

    Args:
        name_to_items: A dictionary mapping save data point names to the corresponding queue items
            holding the prediction tensors and the information for the manifest file.
        output_dir: The directory where the embedding tensors & manifest will be saved.
        manifest_manager: The manifest manager instance to update the manifest file.
    """
    for save_name, entry in name_to_items.items():
        if len(entry.items) > 0:
            save_path = os.path.join(output_dir, save_name)
            is_first_save = entry.save_count == 0
            if is_first_save:
                _, target, input_name, _, split, metadata = QUEUE_ITEM(*entry.items[0])
                target = torch.load(io.BytesIO(target.getbuffer()), map_location="cpu").item()
                manifest_manager.update(input_name, save_name, target, split, metadata)

            prediction_buffers = [item.prediction_buffer for item in entry.items]
            _save_predictions(prediction_buffers, save_path, is_first_save)
            name_to_items[save_name].save_count += 1
            name_to_items[save_name].items = []

    return name_to_items


def _save_predictions(
    prediction_buffers: List[io.BytesIO], save_path: str, is_first_save: bool
) -> None:
    """Saves the embedding tensors as list to .pt files.

    If it's not the first save to this save_path, the new predictions are appended to
    the existing ones and saved to the same file.

    Example use-case: Save all patch embeddings corresponding to the same WSI to a single file.
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
