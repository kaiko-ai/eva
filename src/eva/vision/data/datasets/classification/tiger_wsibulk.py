"""tiger_wsibulk dataset class."""

import functools
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import tifffile as tiff
import torch
from typing_extensions import override

from eva.core.utils.progress_bar import tqdm
from eva.vision.data.datasets import _validators, tiger
from eva.vision.data.wsi.patching import PatchCoordinates, samplers


class TIGERWsiBulk(tiger.TIGERBase):
    """Dataset class for the TIGER tumor detection task.

    Splits a slide-level WSI into multiple different patch level samples,
    dynmaically assigning them labels based on their overlaps with a binary mask.
    """

    _expected_dataset_lengths: Dict[str | None, int] = {
        "train": 65,
        "val": 13,
        "test": 15,
        None: 93,
    }
    """Represents the expected numbers of WSIs in the dataset for validation. 
    Can be overridden for unit tests"""

    _tumor_mask_threshold: float = 0.5
    """ Proportion of the patch that needs to be covered by the mask in order for it to
        be annotated as a "tumor" (1)"""

    _target_mpp: float = 0.5
    """Microns per pixel, in this case stating that a pixel covers 0.5 microns per pixel
    Set as a constant in this implementation to ensure no mis-matches with the binary mask"""

    def __init__(
        self,
        root: str,
        sampler: samplers.Sampler,
        embeddings_dir: str,
        **kwargs,
    ) -> None:
        """Initializes dataset.

        Args:
            root: Root directory of the dataset.
            sampler: The sampler to use for sampling patch coordinates.
            embeddings_dir: Directory where the patch data is stored. Used for annotations.
            kwargs: Key-word arguments from the base class.
        """
        self._embeddings_dir = embeddings_dir
        super().__init__(root=root, sampler=sampler, **kwargs)

    @functools.cached_property
    def annotations(self) -> Dict[str, int]:
        """Builds per-patch labels from the coords CSV files and mask .tif images.

        Returns:
            A dict: { "img_name-patch_index": label }
        """
        annotations = {}

        csv_folder = os.path.normpath(self._embeddings_dir)

        split_to_csv = {
            split: os.path.join(csv_folder, f"coords_{split}.csv")
            for split in ["train", "val", "test"]
        }

        splits_to_load = (
            [self._split] if self._split in ["train", "val", "test"] else ["train", "val", "test"]
        )

        for split in splits_to_load:
            csv_path = split_to_csv[split]
            df = pd.read_csv(csv_path)
            n_rows = len(df)

            for row in tqdm(df.itertuples(index=False), total=n_rows, desc=f"[{split}]"):

                file_name = row.file

                coords = PatchCoordinates(**row._asdict())

                annotations.update(
                    self._process_patch_coordinates(file_name, coords, self._tumor_mask_threshold)
                )

        return annotations

    def _process_patch_coordinates(
        self, file: str, coords: PatchCoordinates, threshold: float
    ) -> dict[str, int]:
        annotations: dict[str, int] = {}
        img_name = Path(file).stem
        patch_w = int(coords.width)
        patch_h = int(coords.height)

        mask_path = os.path.join(self._root, "annotations-tumor-bulk", "masks", f"{img_name}.tif")
        mask = tiff.imread(mask_path)

        for idx, (x, y) in enumerate(coords.x_y):
            patch_region = mask[y : y + patch_h, x : x + patch_w]
            tumor_fraction = np.mean(patch_region > 0)
            label = 1 if tumor_fraction > threshold else 0
            key = f"{img_name}-{idx}"
            annotations[key] = label

        del mask
        return annotations

    @override
    def prepare_data(self) -> None:
        _validators.check_dataset_exists(self._root, False)

    @override
    def validate(self) -> None:
        _validators.check_number_of_files(
            self._file_paths, self._expected_dataset_lengths[self._split], self._split
        )

    @override
    def load_target(self, index: int) -> torch.Tensor:

        metadata = self.load_metadata(index)

        slide_idx = metadata["slide_idx"]
        patch_idx = metadata["patch_idx"]

        file_path = self._file_paths[slide_idx]
        slide_name = Path(file_path).stem
        key = f"{slide_name}-{patch_idx}"
        label = self.annotations[key]

        return torch.tensor(label, dtype=torch.int64)
