"""TotalSegmentator dataset class for multilabel classification."""

import os
from pathlib import Path
from typing import Callable, List, Literal, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from torchvision.datasets.utils import download_url, extract_archive
from typing_extensions import override

from eva.vision.data.datasets import _utils, structs
from eva.vision.data.datasets.classification import base
from eva.vision.utils import io


class TotalSegmentatorClassification(base.ImageClassification):
    """TotalSegmentator dataset class for multilabel classification.

    Create the multi-label classification dataset for the TotalSegmentator data.
    """

    resources: List[structs.DownloadResource] = [
        structs.DownloadResource(
            filename="Totalsegmentator_dataset_v201.zip",
            url="https://zenodo.org/records/10047263/files/Totalsegmentator_dataset_small_v201.zip",
        ),
    ]
    # TODO: switch to use complete instead of small dataset:
    #  - url="https://zenodo.org/records/10047292/files/Totalsegmentator_dataset_v201.zip"

    train_index_ranges: List[Tuple[int, int]] = [(0, 720)]
    """Train range indices."""

    val_index_ranges: List[Tuple[int, int]] = [(720, 880)]
    """Validation range indices."""

    test_index_ranges: List[Tuple[int, int]] = [(880, 1040)]
    """Test range indices."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"],
        download: bool = False,
        image_transforms: Callable | None = None,
        target_transforms: Callable | None = None,
        n_slices_per_image: int = 10,
    ) -> None:
        """Initialize dataset.

        Args:
            root: Path to the root directory of the dataset. The dataset will
                be downloaded and extracted here, if it does not already exist.
            split: Dataset split to use. If None, the entire dataset is used.
            download: Whether to download the data for the specified split.
                Note that the download will be executed only by additionally
                calling the :meth:`prepare_data` method and if the data does not
                exist yet on disk.
            image_transforms: A function/transform that takes in an image
                and returns a transformed version.
            target_transforms: A function/transform that takes in the target
                and transforms it.
            n_slices_per_image: Number of 2D slices to be sampled per 3D image.
        """
        super().__init__(
            image_transforms=image_transforms,
            target_transforms=target_transforms,
        )

        self._root = root
        self._split = split
        self._download = download
        self._n_slices_per_image = n_slices_per_image

        self._samples = []
        self._data: pd.DataFrame
        self._path_key, self._split_key = "path", "split"
        self._manifest_path = os.path.join(self._root, "manifest.csv")
        self._classes = []

    @property
    def dataset_path(self) -> str:
        """Returns the path of the image data of the dataset."""
        return os.path.join(self._root, "Totalsegmentator_dataset_v201")

    @override
    def prepare_data(self) -> None:
        if self._download:
            self._download_dataset()
        if not os.path.isdir(os.path.join(self._root, "Totalsegmentator_dataset_v201")):
            logger.info("Extracting archive ...")
            extract_archive(
                from_path=os.path.join(self._root, "Totalsegmentator_dataset_v201.zip"),
                to_path=os.path.join(self._root, "Totalsegmentator_dataset_v201"),
            )
        self._classes = self._get_classes()

    @override
    def setup(self) -> None:
        self._samples = self._load_samples()
        self._indices = self._make_indices()

    @override
    def load_image(self, index: int) -> np.ndarray:
        image_path, ct_slice = self._samples[index]
        return io.read_nifti_slice_from_image_with_unknown_dimension(
            image_path,
            ct_slice,
            self._n_slices_per_image,
        )

    @override
    def load_target(self, index: int) -> np.ndarray:
        image_path, ct_slice = self._samples[index]
        masks_path = Path(image_path).parents[0] / "segmentations"
        labels = []
        for class_ in self._classes:
            mask_image = io.read_nifti_slice_from_image_with_unknown_dimension(
                str(masks_path / f"{class_}.nii.gz"), ct_slice, self._n_slices_per_image
            )
            labels.append(int(mask_image[:, :].max()))
        return np.asarray(labels, dtype=np.int64)

    @override
    def __len__(self) -> int:
        return len(self._data)

    @override
    def filename(self, index: int) -> str:
        image_path, _ = self._samples[self._indices[index]]
        return os.path.relpath(image_path, self.dataset_path)

    def _download_dataset(self) -> None:
        """Downloads the dataset."""
        os.makedirs(self._root, exist_ok=True)
        for resource in self.resources:
            download_url(
                resource.url,
                root=self._root,
                filename=resource.filename,
                md5=resource.md5,
            )

    def _get_classes(self) -> List[str]:
        """Returns the list with names of the dataset names."""
        classes = [
            f.split(".")[0]
            for f in sorted(
                os.listdir(
                    os.path.join(self._root, "Totalsegmentator_dataset_v201/s0011/segmentations")
                )
            )
        ]
        if len(classes) == 0:
            raise ValueError("No classes found in the dataset.")
        return classes

    def _load_samples(self) -> List[str]:
        """Loads the paths of the samples in the dataset."""
        samples = []
        for image in Path(self._root).glob("**/*ct.nii.gz"):
            for i in range(self._n_slices_per_image):
                samples.append((str(image), i))
        return samples

    def _make_indices(self) -> List[int]:
        """Builds the dataset indices for the specified split."""
        split_index_ranges = {
            "train": self.train_index_ranges,
            "val": self.val_index_ranges,
            "test": self.test_index_ranges,
            None: [(0, 1040)],
        }
        index_ranges = split_index_ranges.get(self._split)
        if index_ranges is None:
            raise ValueError("Invalid data split. Use 'train', 'val', 'test' or `None`.")

        return _utils.ranges_to_indices(index_ranges)
