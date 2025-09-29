"""Text prediction writer callbacks."""

import abc
import os
from typing import Any, Dict, List, Literal, Sequence, Tuple, TypedDict

import lightning.pytorch as pl
import pandas as pd
import torch
import torch.distributed as dist
from lightning.pytorch import callbacks
from torch import nn
from typing_extensions import NotRequired, override

from eva.core.models.modules import utils as module_utils
from eva.core.utils import distributed as dist_utils
from eva.language.models.typings import TextBatch
from eva.language.utils.text import messages as message_utils


class ManifestEntry(TypedDict):
    """A single entry in the manifest file."""

    prediction: str
    """The predicted text."""

    target: str
    """The ground truth text."""

    text: NotRequired[str]
    """The input text data."""

    split: NotRequired[str]
    """The dataset split (e.g. train, val, test)."""


class TextPredictionWriter(callbacks.BasePredictionWriter, abc.ABC):
    """Callback for writing generated text predictions to disk."""

    def __init__(
        self,
        output_dir: str,
        model: nn.Module,
        dataloader_idx_map: Dict[int, str] | None = None,
        metadata_keys: List[str] | None = None,
        include_input: bool = True,
        overwrite: bool = False,
        apply_postprocess: bool = False,
        save_format: Literal["jsonl", "parquet", "csv"] = "jsonl",
    ) -> None:
        """Initializes a new callback.

        Args:
            output_dir: The directory where the embeddings will be saved.
            model: The model instance used to generate the predictions.
            dataloader_idx_map: A dictionary mapping dataloader indices to their respective
                names (e.g. train, val, test).
            metadata_keys: An optional list of keys to extract from the batch metadata and store
                as additional columns in the manifest file.
            include_input: Whether to include the original input text messages in the output.
            overwrite: Whether to overwrite if embeddings are already present in the specified
                output directory. If set to `False`, an error will be raised if embeddings are
                already present (recommended).
            apply_postprocess: Whether to apply the postprocesses specified in the model module.
            save_format: The file format to use for saving the manifest file with the predictions.
        """
        super().__init__()
        self.output_dir = output_dir
        self.model = model
        self.dataloader_idx_map = dataloader_idx_map or {}
        self.metadata_keys = metadata_keys
        self.include_input = include_input
        self.overwrite = overwrite
        self.apply_postprocess = apply_postprocess
        self.save_format = save_format

        self._manifest_path = os.path.join(self.output_dir, f"manifest.{self.save_format}")
        self._data: List[ManifestEntry] = []
        self._is_rank_zero: bool = False

    @override
    def on_predict_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._is_rank_zero = trainer.is_global_zero

        if self._is_rank_zero:
            self._check_if_exists()

        self.model = self.model.to(pl_module.device)
        self.model.eval()

    @override
    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: Any,
        batch_indices: Sequence[int],
        batch: TextBatch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        text_batch, target_batch, metadata_batch = self._unpack_batch(batch)
        has_target = target_batch is not None
        split = self.dataloader_idx_map.get(dataloader_idx, "")

        prediction_batch = self._get_predictions(batch)

        target_batch, prediction_batch = self._apply_postprocess(
            pl_module, target_batch, prediction_batch
        )

        for i in range(len(batch_indices)):
            entry: ManifestEntry = {
                "prediction": str(prediction_batch[i]),
                "target": str(target_batch[i]) if has_target else "",
                "split": split if split else "",
            }
            if self.include_input:
                entry["text"] = message_utils.serialize(text_batch[i])

            if self.metadata_keys is not None and metadata_batch is not None:
                for key in self.metadata_keys:
                    entry[key] = metadata_batch[key][i]

            self._data.append(entry)

    @override
    def on_predict_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Saves the gathered predictions to a manifest file."""
        if dist_utils.is_distributed():
            dist.barrier()
            data = self._gather_data_from_ranks()
        else:
            data = self._data

        if self._is_rank_zero:
            df = pd.DataFrame(data)

            match self.save_format:
                case "jsonl":
                    df.to_json(self._manifest_path, orient="records", lines=True)
                case "parquet":
                    df.to_parquet(self._manifest_path, index=False)
                case "csv":
                    df.to_csv(self._manifest_path, index=False)
                case _:
                    raise ValueError(f"Unsupported save format: {self.save_format}")

    def _gather_data_from_ranks(self) -> List[ManifestEntry]:
        world_size = dist.get_world_size()
        gathered: List[List[ManifestEntry] | None] = [None] * world_size
        dist.all_gather_object(gathered, self._data)
        return [row for shard in gathered for row in (shard or [])]

    def _get_predictions(self, batch: TextBatch) -> List[str]:
        with torch.no_grad():
            output = self.model(batch)

        if (
            not isinstance(output, dict)
            or "generated_text" not in output
            or not all(isinstance(p, str) for p in output["generated_text"])
        ):
            raise ValueError(
                f"A dictionary with 'generated_text' key is expected, got {type(output)}"
            )

        return output["generated_text"]

    def _check_if_exists(self) -> None:
        """Checks if the output directory already exists and if it should be overwritten."""
        os.makedirs(self.output_dir, exist_ok=True)
        if os.path.exists(self._manifest_path) and not self.overwrite:
            raise FileExistsError(
                f"The specified output directory already exists: {self.output_dir}. This "
                "either means that the predictions have been computed before or that a "
                "wrong output directory is being used."
            )

    def _apply_postprocess(
        self, pl_module: pl.LightningModule, targets: Any, predictions: Any
    ) -> Tuple[List[Any], List[Any]]:
        def _to_list(data: Any) -> List[Any]:
            if isinstance(data, torch.Tensor):
                return data.cpu().tolist()
            return data

        if self.apply_postprocess and hasattr(pl_module, "postprocess"):
            if (
                isinstance(pl_module.postprocess, module_utils.BatchPostProcess)
                and pl_module.postprocess.predictions_transforms is not None
            ):
                outputs = {"targets": targets, "predictions": predictions}
                pl_module.postprocess(outputs)
                targets, predictions = outputs["targets"], outputs["predictions"]

        return _to_list(targets), _to_list(predictions)

    def _unpack_batch(self, batch: TextBatch) -> Tuple[list, list | None, dict | None]:
        text_batch, target_batch, metadata_batch = TextBatch(*batch)
        return text_batch, target_batch, metadata_batch
