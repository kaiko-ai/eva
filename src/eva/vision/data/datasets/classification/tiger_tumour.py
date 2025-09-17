"""Tiger_tumour dataset class."""

import ast
import functools
import glob
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, NamedTuple, Tuple

import numpy as np
import pandas as pd
import tifffile as tiff
import torch
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional
from typing_extensions import override

from eva.core.utils.progress_bar import tqdm
from eva.vision.data.datasets import _validators, vision, wsi
from eva.vision.data.wsi.patching import samplers


class TIGERTumour(wsi.MultiWsiDataset, vision.VisionDataset[tv_tensors.Image, torch.Tensor]):
    """Dataset class for the TIL tumour detection task."""

    class ImageRow(NamedTuple):
        """Represents the patch coordinates of one WSI."""

        file: str
        x_y: str
        width: int
        height: int
        level_idx: int

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
            width: Width of the patches to be extracted, in pixels.
            height: Height of the patches to be extracted, in pixels.
            target_mpp: Target microns per pixel (mpp) for the patches.
            backend: The backend to use for reading the whole-slide images.
            image_transforms: Transforms to apply to the extracted image patches.
            coords_path: File path to save the patch coordinates as .csv.
            seed: Random seed for reproducibility.
            n_patches: Number of patches sampled
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
    def annotations(self) -> Dict[str, int]:
        """Builds per-patch labels from the coords CSV files and mask .tif images.

        Returns:
            A dict: { "img_name-patch_index": label }
        """
        annotations = {}

        # Proportion of the patch that needs to be covered by the mask in order for it to
        # be annotated as a "tumor" (1)
        THRESHOLD = 0.5

        main_dir = os.path.dirname(os.path.dirname(self._root))
        csv_folder = os.path.join(main_dir, "embeddings", "dino_vits16", "tiger_tumour")

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

            print(f"Annotating split '{split}' with {n_rows} images...")

            for row in tqdm(df.itertuples(index=False), total=n_rows, desc=f"[{split}]"):
                image_row = TIGERTumour.ImageRow(*row)
                annotations.update(self._process_image_row(image_row, THRESHOLD))

        return annotations

    def _process_image_row(self, row: ImageRow, threshold: float) -> dict[str, int]:
        annotations: dict[str, int] = {}
        img_name = Path(row.file).stem
        patch_coords = ast.literal_eval(row.x_y)
        patch_w = int(row.width)
        patch_h = int(row.height)

        mask_path = os.path.join(self._root, "annotations-tumor-bulk", "masks", f"{img_name}.tif")
        mask = tiff.imread(mask_path)

        for idx, (x, y) in enumerate(patch_coords):
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
        expected_n_files = {"train": 65, "val": 13, "test": 15, None: 93}
        _validators.check_number_of_files(
            self._file_paths, expected_n_files[self._split], self._split
        )

    @override
    def __getitem__(self, index: int) -> Tuple[tv_tensors.Image, torch.Tensor, Dict[str, Any]]:
        return vision.VisionDataset.__getitem__(self, index)

    @override
    def load_data(self, index: int) -> tv_tensors.Image:
        image_array = wsi.MultiWsiDataset.__getitem__(self, index)
        return functional.to_image(image_array)

    @override
    def load_target(self, index: int) -> torch.Tensor:
        slide_idx = index // self._n_patches
        patch_idx = index % self._n_patches

        file_path = self._file_paths[slide_idx]
        slide_name = Path(file_path).stem
        key = f"{slide_name}-{patch_idx}"
        label = self.annotations[key]

        return torch.tensor(label, dtype=torch.int64)

    @override
    def load_metadata(self, index: int) -> Dict[str, Any]:
        return wsi.MultiWsiDataset.load_metadata(self, index)

    def _load_file_paths(self, split: Literal["train", "val", "test"] | None = None) -> List[str]:
        """Loads the file paths of WSIs from wsibulk/images.

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

        return [os.path.relpath(path, self._root) for path in selected_paths]
