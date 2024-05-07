"""WSI classification dataset."""

import os
from typing import Callable, Dict

import pandas as pd
import torch
from typing_extensions import override

from eva.core.models.modules.typings import DATA_SAMPLE
from eva.vision.data.datasets import vision, wsi
from eva.vision.data.wsi.patching import samplers
from eva.vision.data.datasets.classification import base


class WsiClassificationDataset(wsi.MultiWsiDataset, base.ImageClassification):
    """A general dataset class for whole-slide image classification using manifest files."""

    default_column_mapping: Dict[str, str] = {
        "path": "path",
        "target": "target",
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
        transforms: Callable | None = None,
        column_mapping: Dict[str, str] = default_column_mapping,
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
            transforms: Transforms to apply to the extracted patch tensors.
            column_mapping: Mapping of the columns in the manifest file.
        """
        self._root = root
        self._manifest_file = manifest_file
        self._width = width
        self._height = height
        self._target_mpp = target_mpp
        self._sampler = sampler
        self._backend = backend
        self._transforms = transforms
        self._column_mapping = column_mapping

        self._manifest = self._load_manifest(os.path.join(self._root, self._manifest_file))
        self._file_paths = self._manifest[self._column_mapping["path"]].tolist()

        super().__init__(
            root=root,
            file_paths=self._file_paths,
            width=width,
            height=height,
            sampler=sampler,
            target_mpp=target_mpp,
            backend=backend,
            transforms=transforms,
        )

    @override
    def filename(self, index: int) -> str:
        full_path = self._manifest.at[self._get_dataset_idx(index), self._column_mapping["path"]]
        return os.path.basename(full_path)

    @override
    def __getitem__(self, index: int) -> DATA_SAMPLE:
        data = super().__getitem__(index)
        target = self._load_target(index)

        return DATA_SAMPLE(data, target)

    def _load_manifest(self, manifest_path: str) -> pd.DataFrame:
        df = pd.read_csv(manifest_path)

        missing_columns = set(self._column_mapping.values()) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in the manifest file: {missing_columns}")

        return df

    def _load_target(self, index: int) -> torch.Tensor:
        target = self._manifest.at[self._get_dataset_idx(index), self._column_mapping["target"]]
        return torch.tensor(target)
