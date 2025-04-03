"""BRACS dataset class."""

import os
from typing import Any, Callable, Dict, List, Literal, Tuple

import torch
from torchvision import tv_tensors
from torchvision.datasets import folder
from torchvision.transforms.v2 import functional
from typing_extensions import override

from eva.vision.data.datasets import _validators, wsi
from eva.vision.data.datasets.classification import base
from eva.vision.data.wsi.patching import samplers


class BRACS(wsi.MultiWsiDataset, base.ImageClassification):
    """Dataset class for BRACS images and corresponding targets."""

    _expected_files: Dict[str, int] = {
        "train": 3657,
        "val": 312,
        "test": 570,
    }
    """Expected number of files for each split."""

    _license: str = "CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)"
    """Dataset license."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"],
        sampler: samplers.Sampler | None = None,
        width: int = 224,
        height: int = 224,
        target_mpp: float = 0.25,
        transforms: Callable | None = None,
    ) -> None:
        """Initializes the dataset.

        Args:
            root: Path to the root directory of the dataset.
            split: Dataset split to use.
            sampler: The sampler to use for sampling patch coordinates.
                If `None`, it will use the ::class::`ForegroundGridSampler` sampler.
            width: Width of the patches to be extracted, in pixels.
            height: Height of the patches to be extracted, in pixels.
            target_mpp: Target microns per pixel (mpp) for the patches.
            transforms: A function/transform which returns a transformed
                version of the raw data samples.
        """
        self._root = root
        self._split = split
        self._path_to_target = self._make_dataset()
        self._file_to_path = {os.path.basename(p): p for p in self._path_to_target.keys()}

        wsi.MultiWsiDataset.__init__(
            self,
            root=root,
            file_paths=sorted(self._path_to_target.keys()),
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
        return ["0_N", "1_PB", "2_UDH", "3_FEA", "4_ADH", "5_DCIS", "6_IC"]

    @property
    @override
    def class_to_idx(self) -> Dict[str, int]:
        return {name: index for index, name in enumerate(self.classes)}

    @override
    def prepare_data(self) -> None:
        _validators.check_dataset_exists(self._root, True)

    @override
    def validate(self) -> None:
        if len(self._path_to_target) != self._expected_files[self._split]:
            raise ValueError(
                f"Expected {self._split} split to have {self._expected_files[self._split]} files, "
                f"but found {len(self._path_to_target)} files."
            )

        _validators.check_dataset_integrity(
            self,
            length=None,
            n_classes=7,
            first_and_last_labels=("0_N", "6_IC"),
        )

    @override
    def __getitem__(self, index: int) -> Tuple[tv_tensors.Image, torch.Tensor, Dict[str, Any]]:
        return base.ImageClassification.__getitem__(self, index)

    @override
    def load_image(self, index: int) -> tv_tensors.Image:
        image_array = wsi.MultiWsiDataset.__getitem__(self, index)
        return functional.to_image(image_array)

    @override
    def load_target(self, index: int) -> torch.Tensor:
        path = self._file_to_path[self.filename(index)]
        return torch.tensor(self._path_to_target[path], dtype=torch.long)

    @property
    def _dataset_path(self) -> str:
        """Returns the full path of dataset directory."""
        return os.path.join(self._root, "BRACS_RoI/latest_version")

    def _make_dataset(self) -> Dict[str, int]:
        """Builds the dataset for the specified split."""
        dataset = folder.make_dataset(
            directory=os.path.join(self._dataset_path, self._split),
            class_to_idx=self.class_to_idx,
            extensions=(".png"),
        )
        return dict(dataset)

    def _print_license(self) -> None:
        """Prints the dataset license."""
        print(f"Dataset license: {self._license}")
