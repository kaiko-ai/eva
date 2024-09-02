"""Segmentation embeddings writer."""

import collections
import io
import os
from typing import List

import torch
from torch import multiprocessing
from typing_extensions import override

from eva.core.callbacks.writers.embeddings import base
from eva.core.callbacks.writers.embeddings._manifest import ManifestManager
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
        manifest_manager = ManifestManager(output_dir, metadata_keys, overwrite)
        counter = collections.defaultdict(lambda: -1)
        while True:
            item = write_queue.get()
            if item is None:
                break

            embeddings_buffer, target_buffer, input_name, save_name, split, metadata = QUEUE_ITEM(
                *item
            )
            counter[save_name] += 1
            save_name = save_name.replace(".pt", f"-{counter[save_name]}.pt")
            target_filename = save_name.replace(".pt", "-mask.pt")

            _save_embedding(embeddings_buffer, save_name, output_dir)
            _save_embedding(target_buffer, target_filename, output_dir)
            manifest_manager.update(input_name, save_name, target_filename, split, metadata)

        manifest_manager.close()

    @override
    def _get_embeddings(self, tensor: torch.Tensor) -> torch.Tensor | List[List[torch.Tensor]]:
        """Returns the embeddings from predictions."""

        def _get_grouped_embeddings(embeddings: List[torch.Tensor]) -> List[List[torch.Tensor]]:
            """Casts a list of multi-leveled batched embeddings to grouped per batch.

            That is, for embeddings to be a list of shape (batch_size, hidden_dim, height, width),
            such as `[(2, 192, 16, 16), (2, 192, 16, 16)]`, to be reshaped as a list of lists of
            per batch multi-level embeddings, thus
            `[ [(192, 16, 16), (192, 16, 16)], [(192, 16, 16), (192, 16, 16)] ]`.
            """
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


def _save_embedding(embeddings_buffer: io.BytesIO, save_name: str, output_dir: str) -> None:
    save_path = os.path.join(output_dir, save_name)
    prediction = torch.load(io.BytesIO(embeddings_buffer.getbuffer()), map_location="cpu")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(prediction, save_path)
