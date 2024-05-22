"""Classification embeddings writer."""

import csv
import io
import os
from typing import Any

import torch
from torch import multiprocessing
from typing_extensions import override

from eva.core.callbacks.writers.embeddings import base
from eva.core.callbacks.writers.embeddings.typings import QUEUE_ITEM


class ClassificationEmbeddingsWriter(base.EmbeddingsWriter):
    """Callback for writing generated embeddings to disk."""

    @staticmethod
    @override
    def _process_write_queue(
        write_queue: multiprocessing.Queue, output_dir: str, overwrite: bool = False
    ) -> None:
        manifest_file, manifest_writer = _init_manifest(output_dir, overwrite)
        while True:
            item = write_queue.get()
            if item is None:
                break

            embeddings_buffer, target_buffer, input_name, save_name, split = QUEUE_ITEM(*item)
            _save_embedding(embeddings_buffer, save_name, output_dir)
            _update_manifest(target_buffer, input_name, save_name, split, manifest_writer)

        manifest_file.close()


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
    target_buffer: io.BytesIO,
    input_name: str,
    save_name: str,
    split: str | None,
    manifest_writer,
) -> None:
    buffer = io.BytesIO(target_buffer.getbuffer())
    target = torch.load(buffer, map_location="cpu")
    manifest_writer.writerow([input_name, save_name, target.item(), split])
