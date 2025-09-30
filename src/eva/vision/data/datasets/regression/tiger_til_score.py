"""Tiger dataset class for regression targets."""

import functools
import os
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from typing_extensions import override

from eva.vision.data.datasets import tiger


class TIGERTILScore(tiger.TIGERBase):
    """Dataset class for regression tasks using the TIGERTILS partition of the TIGER dataset.

    Predicts TIL scores, i.e. the proportion of the cell infiltrated by TILs.
    """

    @functools.cached_property
    def annotations(self) -> Dict[str, float]:
        """Loads per-slide regression targets from a CSV file.

        Expected CSV format:
            image-id,tils-score
            103S,0.70
            ...
        """
        targets_csv_path = os.path.join(self._root, "tiger-til-scores-wsitils.csv")

        if not os.path.isfile(targets_csv_path):
            raise FileNotFoundError(f"Targets CSV file not found at: {targets_csv_path}")

        df = pd.read_csv(targets_csv_path)
        if not {"image-id", "tils-score"} <= set(df.columns):
            raise ValueError("targets_csv must contain 'image-id' and 'tils-score' columns.")

        return {str(row["image-id"]): float(row["tils-score"]) for _, row in df.iterrows()}

    @override
    def load_target(self, index: int) -> torch.Tensor:
        metadata = self.load_metadata(index=index)
        slide_idx = metadata["slide_idx"]
        file_path = self._file_paths[slide_idx]
        slide_name = Path(file_path).stem

        target_value = self.annotations[slide_name]
        tensor = torch.tensor([target_value], dtype=torch.float32)
        return tensor
