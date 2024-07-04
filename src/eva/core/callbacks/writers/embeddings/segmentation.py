"""Segmentation embeddings writer."""

import csv
import io
import os
from typing import Any, List

import torch
from torch import multiprocessing
from typing_extensions import override

from eva.core.callbacks.writers.embeddings import base
from eva.core.callbacks.writers.embeddings.typings import QUEUE_ITEM


class SegmentationEmbeddingsWriter(base.EmbeddingsWriter):
    """Callback for writing generated embeddings to disk."""

    @staticmethod
    @override
    def _process_write_queue(
        write_queue: multiprocessing.Queue,
        output_dir: str,
        metadata_keys: List[str],
        save_every_n: int,
        overwrite: bool = False,
    ) -> None:
        manifest_file, manifest_writer = _init_manifest(output_dir, overwrite)
        while True:
            item = write_queue.get()
            if item is None:
                break

            embeddings_buffer, target_buffer, input_name, save_name, split, _ = QUEUE_ITEM(*item)
            target_filename = save_name.replace(".pt", "-mask.pt")
            _save_embedding(embeddings_buffer, save_name, output_dir)
            _save_embedding(target_buffer, target_filename, output_dir)
            _update_manifest(target_filename, input_name, save_name, split, manifest_writer)

        manifest_file.close()

    @override
    def _get_embeddings(self, tensor: torch.Tensor) -> torch.Tensor | List[torch.Tensor]:
        """Returns the embeddings from predictions."""

        def _get_grouped_embeddings(embeddings: List[torch.Tensor]) -> List[List[torch.Tensor]]:
            """"""
            batch_size = embeddings[0].shape[0]
            grouped_embeddings = []
            for batch_idx in range(batch_size):
                batch_list = [layer_embeddings[batch_idx] for layer_embeddings in embeddings]
                grouped_embeddings.append(batch_list)
            return grouped_embeddings

        embeddings = self._backbone(tensor) if self._backbone else tensor
        if isinstance(embeddings, list):
            embeddings = _get_grouped_embeddings(embeddings)
        return embeddings


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


def _save_embedding(embeddings_buffer: io.BytesIO, save_name: str, output_dir: str) -> None:
    save_path = os.path.join(output_dir, save_name)
    prediction = torch.load(io.BytesIO(embeddings_buffer.getbuffer()), map_location="cpu")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(prediction, save_path)


def _update_manifest(
    target_filename: str,
    input_name: str,
    save_name: str,
    split: str | None,
    manifest_writer,
) -> None:
    manifest_writer.writerow([input_name, save_name, target_filename, split])
