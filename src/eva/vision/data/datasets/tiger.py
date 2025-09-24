"""Abstract base class for TIGER datasets spanning different task types."""

import abc
import glob
import os
from typing import Any, Callable, Dict, List, Literal, Tuple

import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional
from typing_extensions import override

from eva.vision.data.datasets import _validators, vision, wsi
from eva.vision.data.wsi.patching import samplers


class TIGERBase(
    wsi.MultiWsiDataset,
    vision.VisionDataset[tv_tensors.Image, torch.Tensor],
    abc.ABC,
):
    """Abstract base class for TIGER datasets spanning different task types."""

    _train_split_ratio: float = 0.7
    _val_split_ratio: float = 0.15

    # target microns per pixel (mpp) for patches.
    _target_mpp: float = 0.5

    def __init__(
        self,
        root: str,
        sampler: samplers.Sampler,
        split: Literal["train", "val", "test"] | None = None,
        width: int = 224,
        height: int = 224,
        backend: str = "openslide",
        image_transforms: Callable | None = None,
        coords_path: str | None = None,
        seed: int = 42,
    ) -> None:
        """Initializes the dataset.

        Args:
            root: Root directory of the dataset.
            sampler: The sampler to use for sampling patch coordinates.
            split: Dataset split to use. If `None`, the entire dataset is used.
            width: Patch width in pixels.
            height: Patch height in pixels.
            backend: WSI reading backend.
            image_transforms: Transforms to apply to patches.
            coords_path: Optional path to save patch coordinates.
            seed: Random seed.
        """
        self._root = root
        self._split = split
        self._width = width
        self._height = height
        self._seed = seed

        wsi.MultiWsiDataset.__init__(
            self,
            root=root,
            file_paths=self._load_file_paths(split),
            width=width,
            height=height,
            sampler=sampler,
            target_mpp=self._target_mpp,
            backend=backend,
            image_transforms=image_transforms,
            coords_path=coords_path,
        )

    @override
    def prepare_data(self) -> None:
        _validators.check_dataset_exists(self._root, False)

    @override
    def __getitem__(self, index: int) -> Tuple[tv_tensors.Image, torch.Tensor, Dict[str, Any]]:
        return vision.VisionDataset.__getitem__(self, index)

    @override
    def load_data(self, index: int) -> tv_tensors.Image:
        image_array = wsi.MultiWsiDataset.__getitem__(self, index)
        return functional.to_image(image_array)

    @override
    def load_metadata(self, index: int) -> Dict[str, Any]:
        return wsi.MultiWsiDataset.load_metadata(self, index)

    @abc.abstractmethod
    def annotations(self) -> Dict[str, Any]:
        """Annotates target data."""
        raise NotImplementedError

    @abc.abstractmethod
    def load_target(self, index: int):
        """Task-specific target loading."""
        raise NotImplementedError

    def _load_file_paths(self, split: Literal["train", "val", "test"] | None = None) -> List[str]:
        """Loads the file paths of WSIs from wsibulk/images.

        Splits are assigned 70% train, 15% val, 15% test by filename sorting.
        """
        image_dir = os.path.join(self._root, "images")
        all_paths = sorted(glob.glob(os.path.join(image_dir, "*.tif")))

        if not all_paths:
            raise FileNotFoundError(f"No .tif files found in {image_dir}")

        rng = np.random.default_rng(self._seed)  # nosec B311
        rng.shuffle(all_paths)

        n_total = len(all_paths)
        n_train = int(n_total * self._train_split_ratio)
        n_val = int(n_total * self._val_split_ratio)

        if split == "train":
            selected_paths = all_paths[:n_train]
        elif split == "val":
            selected_paths = all_paths[n_train : n_train + n_val]
        elif split == "test":
            selected_paths = all_paths[n_train + n_val :]
        elif split is None:
            selected_paths = all_paths

        return [os.path.relpath(path, self._root) for path in selected_paths]
