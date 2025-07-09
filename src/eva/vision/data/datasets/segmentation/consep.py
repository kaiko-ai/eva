"""CoNSeP dataset."""

import glob
import os
from typing import Any, Callable, Dict, List, Literal, Tuple

import numpy as np
import numpy.typing as npt
import torch
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional
from typing_extensions import override

from eva.vision.data.datasets import _validators, vision, wsi
from eva.vision.data.datasets.segmentation import _utils
from eva.vision.data.wsi.patching import samplers
from eva.vision.utils import io


class CoNSeP(wsi.MultiWsiDataset, vision.VisionDataset[tv_tensors.Image, tv_tensors.Mask]):
    """Dataset class for CoNSeP semantic segmentation task.

    As in [1], we combine classes 3 (healthy epithelial) & 4 (dysplastic/malignant epithelial)
    into the epithelial class and 5 (fibroblast), 6 (muscle) & 7 (endothelial) into
    the spindle-shaped class.

    [1] Graham, Simon, et al. "Hover-net: Simultaneous segmentation and classification of
        nuclei in multi-tissue histology images." https://arxiv.org/abs/1802.04712
    """

    _expected_dataset_lengths: Dict[str | None, int] = {
        "train": 27,
        "val": 14,
        None: 41,
    }
    """Expected dataset lengths for the splits and complete dataset."""

    def __init__(
        self,
        root: str,
        sampler: samplers.Sampler | None = None,
        split: Literal["train", "val"] | None = None,
        width: int = 250,
        height: int = 250,
        target_mpp: float = 0.25,
        transforms: Callable | None = None,
    ) -> None:
        """Initializes the dataset.

        Args:
            root: Root directory of the dataset.
            sampler: The sampler to use for sampling patch coordinates.
                If `None`, it will use the ::class::`ForegroundGridSampler` sampler.
            split: Dataset split to use. If `None`, the entire dataset is used.
            width: Width of the patches to be extracted, in pixels.
            height: Height of the patches to be extracted, in pixels.
            target_mpp: Target microns per pixel (mpp) for the patches.
            transforms: Transforms to apply to the extracted image & mask patches.
        """
        self._split = split
        self._root = root

        self.datasets: List[wsi.WsiDataset]  # type: ignore

        wsi.MultiWsiDataset.__init__(
            self,
            root=root,
            file_paths=self._load_file_paths(split),
            width=width,
            height=height,
            sampler=sampler or samplers.ForegroundGridSampler(max_samples=25),
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
            length=None,
            n_classes=5,
            first_and_last_labels=((self.classes[0], self.classes[-1])),
        )
        n_expected = self._expected_dataset_lengths[None]
        if len(self._file_paths) != n_expected:
            raise ValueError(
                f"Expected {n_expected} images, found {len(self._file_paths)} in {self._root}."
            )

    @override
    def __getitem__(self, index: int) -> Tuple[tv_tensors.Image, tv_tensors.Mask, Dict[str, Any]]:
        return vision.VisionDataset.__getitem__(self, index)

    @override
    def load_data(self, index: int) -> tv_tensors.Image:
        image_array = wsi.MultiWsiDataset.__getitem__(self, index)
        return functional.to_image(image_array)

    @override
    def load_target(self, index: int) -> tv_tensors.Mask:
        path = self._get_mask_path(index)
        mask = np.array(io.read_mat(path)["type_map"])
        mask_patch = _utils.extract_mask_patch(mask, self, index)
        mask_patch = self._map_classes(mask_patch)
        mask_tensor = tv_tensors.Mask(mask_patch, dtype=torch.int64)  # type: ignore[reportCallIssue]
        return self._image_transforms(mask_tensor) if self._image_transforms else mask_tensor

    @override
    def load_metadata(self, index: int) -> Dict[str, Any]:
        (x, y), width, height = _utils.get_coords_at_index(self, index)
        return {"coords": f"{x},{y},{width},{height}"}

    def _load_file_paths(self, split: Literal["train", "val"] | None = None) -> List[str]:
        """Loads the file paths of the corresponding dataset split."""
        paths = list(glob.glob(os.path.join(self._root, "**/Images/*.png"), recursive=True))
        if split is not None:
            split_to_folder = {"train": "Train", "val": "Test"}
            paths = filter(lambda p: split_to_folder[split] == p.split("/")[-3], paths)

        return sorted(paths)

    def _get_mask_path(self, index: int) -> str:
        """Returns the path to the mask file corresponding to the patch at the given index."""
        filename = self.filename(index).split(".")[0]
        mask_dir = "Train" if filename.startswith("train") else "Test"
        return os.path.join(self._root, mask_dir, "Labels", f"{filename}.mat")

    def _map_classes(self, array: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Summarizes classes 3 & 4, and 5, 6."""
        array = np.where(array == 4, 3, array)
        array = np.where(array > 4, 4, array)
        return array
