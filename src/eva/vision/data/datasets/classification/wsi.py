"""WSI classification dataset."""

import os
from typing import Any, Callable, Dict, Literal, Tuple

import numpy as np
import pandas as pd
import torch
from torchvision import tv_tensors
from typing_extensions import override

from eva.vision.data.datasets import vision, wsi
from eva.vision.data.wsi.patching import samplers


class WsiClassificationDataset(
    wsi.MultiWsiDataset, vision.VisionDataset[tv_tensors.Image, torch.Tensor]
):
    """A general dataset class for whole-slide image classification using manifest files."""

    default_column_mapping: Dict[str, str] = {
        "path": "path",
        "target": "target",
        "split": "split",
    }

    def __init__(
        self,
        root: str,
        manifest_file: str,
        width: int,
        height: int,
        target_mpp: float,
        sampler: samplers.Sampler,
        backend: str = "openslide",
        split: Literal["train", "val", "test"] | None = None,
        image_transforms: Callable | None = None,
        column_mapping: Dict[str, str] = default_column_mapping,
        coords_path: str | None = None,
    ):
        """Initializes the dataset.

        Args:
            root: Root directory of the dataset.
            manifest_file: The path to the manifest file, relative to
                the `root` argument. The `path` column is expected to contain
                relative paths to the whole-slide images.
            width: Width of the patches to be extracted, in pixels.
            height: Height of the patches to be extracted, in pixels.
            target_mpp: Target microns per pixel (mpp) for the patches.
            sampler: The sampler to use for sampling patch coordinates.
            backend: The backend to use for reading the whole-slide images.
            split: The split of the dataset to load.
            image_transforms: Transforms to apply to the extracted image patches.
            column_mapping: Mapping of the columns in the manifest file.
            coords_path: File path to save the patch coordinates as .csv.
        """
        self._split = split
        self._column_mapping = self.default_column_mapping | column_mapping
        self._manifest = self._load_manifest(os.path.join(root, manifest_file))

        wsi.MultiWsiDataset.__init__(
            self,
            root=root,
            file_paths=self._manifest[self._column_mapping["path"]].tolist(),
            width=width,
            height=height,
            sampler=sampler,
            target_mpp=target_mpp,
            backend=backend,
            image_transforms=image_transforms,
            coords_path=coords_path,
        )

    @override
    def filename(self, index: int) -> str:
        path = self._manifest.at[self._get_dataset_idx(index), self._column_mapping["path"]]
        return os.path.basename(path) if os.path.isabs(path) else path

    @override
    def __getitem__(self, index: int) -> Tuple[tv_tensors.Image, torch.Tensor, Dict[str, Any]]:
        return vision.VisionDataset.__getitem__(self, index)

    @override
    def load_data(self, index: int) -> tv_tensors.Image:
        return wsi.MultiWsiDataset.__getitem__(self, index)

    @override
    def load_target(self, index: int) -> np.ndarray:
        target = self._manifest.at[self._get_dataset_idx(index), self._column_mapping["target"]]
        return np.asarray(target)

    @override
    def load_metadata(self, index: int) -> Dict[str, Any]:
        return wsi.MultiWsiDataset.load_metadata(self, index)

    def _load_manifest(self, manifest_path: str) -> pd.DataFrame:
        df = pd.read_csv(manifest_path)

        missing_columns = set(self._column_mapping.values()) - set(df.columns)
        if self._split is None:
            missing_columns = missing_columns - {self._column_mapping["split"]}
        if missing_columns:
            raise ValueError(f"Missing columns in the manifest file: {missing_columns}")

        if self._split is not None:
            df = df.loc[df[self._column_mapping["split"]] == self._split]

        return df.reset_index(drop=True)
