"""PANDA dataset class."""

import glob
import os
from typing import Any, Callable, Dict, List, Literal, Tuple

import numpy as np
import numpy.typing as npt
import scipy.io
import torch
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional
from typing_extensions import override

from eva.vision.data.datasets import _validators, wsi
from eva.vision.data.datasets.segmentation import base
from eva.vision.data.wsi.patching import samplers


class CoNSeP(wsi.MultiWsiDataset, base.ImageSegmentation):
    """Dataset class for CoNSeP semantic segmentation task.

    We combine classes 3 (healthy epithelial) & 4 (dysplastic/malignant epithelial)
    into the epithelial class and 5 (fibroblast), 6 (muscle) & 7 (endothelial) into
    the spindle-shaped class.
    """

    _expected_dataset_lengths: Dict[str | None, int] = {
        "train": 27,
        "val": 14,
        None: 41,
    }

    def __init__(
        self,
        root: str,
        sampler: samplers.Sampler,
        split: Literal["train", "val"] | None = None,
        width: int = 224,
        height: int = 224,
        target_mpp: float = 0.25,
        transforms: Callable | None = None,
        seed: int = 42,
    ) -> None:
        """Initializes the dataset.

        Args:
            root: Root directory of the dataset.
            sampler: The sampler to use for sampling patch coordinates.
            split: Dataset split to use. If `None`, the entire dataset is used.
            width: Width of the patches to be extracted, in pixels.
            height: Height of the patches to be extracted, in pixels.
            target_mpp: Target microns per pixel (mpp) for the patches.
            backend: The backend to use for reading the whole-slide images.
            transforms: Transforms to apply to the extracted image & mask patches.
            seed: Random seed for reproducibility.
        """
        self._split = split
        self._root = root
        self._seed = seed

        self.datasets: List[wsi.WsiDataset]  # type: ignore

        wsi.MultiWsiDataset.__init__(
            self,
            root=root,
            file_paths=self._load_file_paths(split),
            width=width,
            height=height,
            sampler=sampler,
            target_mpp=target_mpp,
            overwrite_mpp=0.25,
            backend="pil",
            image_transforms=transforms,
        )

    @property
    @override
    def classes(self) -> List[str]:
        return [
            "background",
            "other",
            "inflammatory",
            "epithelial",
            "spindle-shaped",
        ]

    @property
    @override
    def class_to_idx(self) -> Dict[str, int]:
        return {label: index for index, label in enumerate(self.classes)}

    @override
    def prepare_data(self) -> None:
        _validators.check_dataset_exists(self._root, True)

        if not os.path.isdir(os.path.join(self._root, "Train")):
            raise FileNotFoundError(f"Train directory not found in {self._root}.")
        if not os.path.isdir(os.path.join(self._root, "Test")):
            raise FileNotFoundError(f"Test directory not found in {self._root}.")

    @override
    def validate(self) -> None:
        _validators.check_dataset_integrity(
            self,
            length=self._expected_dataset_lengths.get(self._split),
            n_classes=4,
            first_and_last_labels=((self.classes[0], self.classes[-1])),
        )

    @override
    def filename(self, index: int) -> str:
        return os.path.basename(self._file_paths[self._get_dataset_idx(index)])

    @override
    def __getitem__(self, index: int) -> Tuple[tv_tensors.Image, tv_tensors.Mask]:
        return base.ImageSegmentation.__getitem__(self, index)

    @override
    def load_image(self, index: int) -> tv_tensors.Image:
        image_array = wsi.MultiWsiDataset.__getitem__(self, index)
        return functional.to_image(image_array)

    @override
    def load_mask(self, index: int) -> tv_tensors.Mask:
        path = self._get_mask_path(index)
        mask = scipy.io.loadmat(path)["type_map"]
        mask_patch = self._read_mask_patch(index, mask)  # type: ignore
        # mask_patch = self._map_classes(mask_patch)
        return tv_tensors.Mask(mask_patch, dtype=torch.int64)  # type: ignore[reportCallIssue]

    def _load_file_paths(self, split: Literal["train", "val"] | None = None) -> List[str]:
        """Loads the file paths of the corresponding dataset split."""
        paths = list(glob.glob(os.path.join(self._root, "**/Images/*.png"), recursive=True))
        n_expected = self._expected_dataset_lengths[None]
        if len(paths) != n_expected:
            raise ValueError(f"Expected {n_expected} images, found {len(paths)} in {self._root}.")

        if split is not None:
            split_to_folder = {"train": "Train", "val": "Test"}
            paths = filter(lambda p: split_to_folder[split] == p.split("/")[-3], paths)

        return sorted(paths)

    def _get_coords(self, index: int) -> Tuple[Tuple[int, int], int, int]:
        """Returns the coordinates ((x,y),width,height) of the patch at the given index."""
        image_index = self._get_dataset_idx(index)
        patch_index = index if image_index == 0 else index - self.cumulative_sizes[image_index - 1]
        coords = self.datasets[image_index]._coords
        return coords.x_y[patch_index], coords.width, coords.height

    def _get_mask_path(self, index):
        """Returns the path to the mask file corresponding to the patch at the given index."""
        filename = self.filename(index).split(".")[0]
        mask_dir = "Train/Labels" if filename.startswith("train") else "Test/Labels"
        return os.path.join(self._root, mask_dir, f"{filename}.mat")

    def _read_mask_patch(self, index: int, mask: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Reads the mask patch at the coordinates corresponding to the given patch index."""
        (x, y), width, height = self._get_coords(index)
        return mask[x : x + width, y : y + height]

    def _map_classes(self, array: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Summarizes classes 3 & 4, and 5, 6."""
        array = np.where(array == 4, 3, array)
        array = np.where(array > 4, 4, array)
        return array
