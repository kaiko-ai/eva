"""Tiger dataset class for regression targets."""

import functools
import glob
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal

import pandas as pd
import torch
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional
from typing_extensions import override

from eva.vision.data.datasets import _validators, vision, wsi
from eva.vision.data.wsi.patching import samplers


class TIGERTILScore(wsi.MultiWsiDataset, vision.VisionDataset[tv_tensors.Image, torch.Tensor]):
    """Dataset class for TIGERBULK regression tasks with per-slide targets."""

    def __init__(
        self,
        root: str,
        sampler: samplers.Sampler,
        split: Literal["train", "val", "test"] | None = None,
        width: int = 224,
        height: int = 224,
        target_mpp: float = 0.5,
        backend: str = "openslide",
        image_transforms: Callable | None = None,
        coords_path: str | None = None,
        seed: int = 42,
        n_patches: int = 200,
    ) -> None:
        """Initializes the dataset.

        Args:
            root: Root directory of the dataset.
            sampler: The sampler to use for sampling patch coordinates.
            split: Dataset split to use. If `None`, the entire dataset is used.
            width: Patch width in pixels.
            height: Patch height in pixels.
            target_mpp: Target microns per pixel (mpp) for patches.
            backend: WSI reading backend.
            image_transforms: Transforms to apply to patches.
            coords_path: Optional path to save patch coordinates.
            seed: Random seed.
            n_patches: Number of patches per slide.
            targets_csv: Path to CSV containing per-slide regression targets.
                        Must have columns: slide_name,target
        """
        self._split = split
        self._root = root
        self._width = width
        self._height = height
        self._target_mpp = target_mpp
        self._seed = seed
        self._n_patches = n_patches

        wsi.MultiWsiDataset.__init__(
            self,
            root=root,
            file_paths=self._load_file_paths(split),
            width=width,
            height=height,
            sampler=sampler,
            target_mpp=target_mpp,
            backend=backend,
            image_transforms=image_transforms,
            coords_path=coords_path,
        )

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
    def prepare_data(self) -> None:
        _validators.check_dataset_exists(self._root, False)

    @override
    def __getitem__(self, index: int):
        return vision.VisionDataset.__getitem__(self, index)

    @override
    def load_data(self, index: int) -> tv_tensors.Image:
        image_array = wsi.MultiWsiDataset.__getitem__(self, index)
        return functional.to_image(image_array)

    @override
    def load_target(self, index: int) -> torch.Tensor:
        slide_idx = index // self._n_patches
        file_path = self._file_paths[slide_idx]
        slide_name = Path(file_path).stem

        target_value = self.annotations[slide_name]
        tensor = torch.tensor([target_value], dtype=torch.float32)
        return tensor

    @override
    def load_metadata(self, index: int) -> Dict[str, Any]:
        return wsi.MultiWsiDataset.load_metadata(self, index)

    def _load_file_paths(self, split: Literal["train", "val", "test"] | None = None) -> List[str]:
        """Loads the file paths of WSIs from wsitils/images.

        Splits are assigned 70% train, 15% val, 15% test by filename sorting.
        """
        image_dir = os.path.join(self._root, "images")

        all_paths = sorted(glob.glob(os.path.join(image_dir, "*.tif")))

        if not all_paths:
            raise FileNotFoundError(f"No .tif files found in {image_dir}")

        n_total = len(all_paths)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)

        if split == "train":
            selected_paths = all_paths[:n_train]
        elif split == "val":
            selected_paths = all_paths[n_train : n_train + n_val]
        elif split == "test":
            selected_paths = all_paths[n_train + n_val :]
        elif split is None:
            selected_paths = all_paths
        else:
            raise ValueError("Invalid split. Use 'train', 'val', 'test', or None.")

        return [os.path.relpath(path, self._root) for path in selected_paths]
