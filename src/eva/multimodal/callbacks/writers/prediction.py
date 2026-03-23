"""Text prediction writer callbacks."""

from typing import Dict, List, Literal, Tuple

from torch import nn
from typing_extensions import override

from eva.language.callbacks import writers
from eva.multimodal.models.typings import TextImageBatch


class TextPredictionWriter(writers.TextPredictionWriter):
    """Callback for writing generated text predictions to disk."""

    def __init__(
        self,
        output_dir: str,
        model: nn.Module,
        dataloader_idx_map: Dict[int, str] | None = None,
        metadata_keys: List[str] | None = None,
        include_input: bool = True,
        overwrite: bool = False,
        save_format: Literal["jsonl", "parquet", "csv"] = "jsonl",
    ) -> None:
        """See docstring of base class."""
        super().__init__(
            output_dir=output_dir,
            model=model,
            dataloader_idx_map=dataloader_idx_map,
            metadata_keys=metadata_keys,
            include_input=include_input,
            overwrite=overwrite,
            save_format=save_format,
        )

    @override
    def _unpack_batch(self, batch: TextImageBatch) -> Tuple[list, list | None, dict | None]:  # type: ignore
        text_batch, _, target_batch, metadata_batch = TextImageBatch(*batch)
        return text_batch, target_batch, metadata_batch
