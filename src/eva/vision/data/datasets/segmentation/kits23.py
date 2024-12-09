"""KiTS23 dataset."""

import functools
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Tuple
from urllib import request

import nibabel as nib
import torch
from torchvision import tv_tensors
from typing_extensions import override

from eva.core.data import splitting
from eva.core.utils import multiprocessing
from eva.core.utils.progress_bar import tqdm
from eva.vision.data.datasets import _utils, _validators
from eva.vision.data.datasets.segmentation import base
from eva.vision.utils import io
from eva.vision.utils.io import nifti


class KiTS23(base.ImageSegmentation):
    """KiTS23 - The 2023 Kidney and Kidney Tumor Segmentation challenge.

    To optimize data loading, the dataset is preprocessed by reorienting the images
    from IPL to LAS and uncompressing them. The reorientation is necessary, because
    loading slices from the first dimension is significantly slower than from the last,
    due to data not being stored in a contiguous manner on disk accross all dimensions.

    Webpage: https://kits-challenge.org/kits23/
    """

    _index_ranges: List[Tuple[int, int]] = [(0, 300), (400, 589)]
    """Dataset index ranges."""

    _train_ratio: float = 0.7
    _val_ratio: float = 0.15
    _test_ratio: float = 0.15
    """Ratios for dataset splits."""

    _expected_dataset_lengths: Dict[str | None, int] = {
        "train": 67582,
        "val": 13751,
        "test": 13888,
        None: 95221,
    }
    """Dataset version and split to the expected size."""

    _sample_every_n_slices: int | None = None
    """The amount of slices to sub-sample per 3D CT scan image."""

    _processed_dir: str = "processed"
    """Directory where the processed data (reoriented to LPS & uncompressed) is stored."""

    _license: str = "CC BY-NC-SA 4.0"
    """Dataset license."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"] | None = None,
        download: bool = False,
        num_workers: int = 10,
        transforms: Callable | None = None,
        seed: int = 8,
    ) -> None:
        """Initialize dataset.

        Args:
            root: Path to the root directory of the dataset. The dataset will
                be downloaded and extracted here, if it does not already exist.
            split: Dataset split to use. If `None`, the entire dataset will be used.
            download: Whether to download the data for the specified split.
                Note that the download will be executed only by additionally
                calling the :meth:`prepare_data` method and if the data does
                not yet exist on disk.
            num_workers: The number of workers to use for preprocessing the dataset.
            transforms: A function/transforms that takes in an image and a target
                mask and returns the transformed versions of both.
            seed: Seed used for generating the dataset splits.
        """
        super().__init__(transforms=transforms)

        self._root = root
        self._split = split
        self._download = download
        self._num_workers = num_workers
        self._seed = seed

        self._indices: List[Tuple[int, int]] = []

    @property
    @override
    def classes(self) -> List[str]:
        return ["background", "kidney", "tumor", "cyst"]

    @functools.cached_property
    @override
    def class_to_idx(self) -> Dict[str, int]:
        return {label: index for index, label in enumerate(self.classes)}

    @property
    def _processed_root(self) -> str:
        return os.path.join(self._root, self._processed_dir)

    @override
    def filename(self, index: int) -> str:
        sample_index, _ = self._indices[index]
        return self._volume_filename(sample_index)

    @override
    def prepare_data(self) -> None:
        if self._download:
            self._download_dataset()
        self._preprocess()

    @override
    def configure(self) -> None:
        self._indices = self._create_indices()

    @override
    def validate(self) -> None:
        _validators.check_dataset_integrity(
            self,
            length=self._expected_dataset_lengths.get(self._split, 0),
            n_classes=4,
            first_and_last_labels=("background", "cyst"),
        )

    @override
    def load_image(self, index: int) -> tv_tensors.Image:
        sample_index, slice_index = self._indices[index]
        volume_path = self._volume_path(sample_index)
        image_array = io.read_nifti(volume_path, slice_index)
        return tv_tensors.Image(image_array.transpose(2, 0, 1), dtype=torch.float32)  # type: ignore[reportCallIssue]

    @override
    def load_mask(self, index: int) -> tv_tensors.Mask:
        sample_index, slice_index = self._indices[index]
        segmentation_path = self._segmentation_path(sample_index)
        semantic_labels = io.read_nifti(segmentation_path, slice_index)
        return tv_tensors.Mask(semantic_labels.squeeze(), dtype=torch.int64)  # type: ignore[reportCallIssue]

    @override
    def load_metadata(self, index: int) -> Dict[str, Any]:
        sample_index, slice_index = self._indices[index]
        return {"case_id": f"{sample_index:05d}", "slice_index": slice_index}

    @override
    def __len__(self) -> int:
        return len(self._indices)

    def _create_indices(self) -> List[Tuple[int, int]]:
        """Builds the dataset indices for the specified split.

        Returns:
            A list of tuples, where the first value indicates the
            sample index which the second its corresponding slice
            index.
        """
        indices = [
            (sample_idx, slide_idx)
            for sample_idx in self._get_split_indices()
            for slide_idx in range(self._get_number_of_slices_per_volume(sample_idx))
            if slide_idx % (self._sample_every_n_slices or 1) == 0
        ]
        return indices

    def _get_split_indices(self) -> List[int]:
        """Builds the dataset indices for the specified split."""
        indices = _utils.ranges_to_indices(self._index_ranges)

        train_indices, val_indices, test_indices = splitting.random_split(
            indices, self._train_ratio, self._val_ratio, self._test_ratio, seed=self._seed
        )
        split_indices_dict = {
            "train": [indices[i] for i in train_indices],
            "val": [indices[i] for i in val_indices],
            "test": [indices[i] for i in test_indices],  # type: ignore
            None: indices,
        }
        if self._split not in split_indices_dict:
            raise ValueError("Invalid data split. Use 'train', 'val', 'test' or `None`.")

        return list(split_indices_dict[self._split])

    def _get_number_of_slices_per_volume(self, sample_index: int) -> int:
        """Returns the total amount of slices of a volume."""
        volume_shape = io.fetch_nifti_shape(self._volume_path(sample_index))
        return volume_shape[-1]

    def _volume_filename(self, sample_index: int) -> str:
        return f"case_{sample_index:05d}/master_{sample_index:05d}.nii"

    def _segmentation_filename(self, sample_index: int) -> str:
        return f"case_{sample_index:05d}/segmentation.nii"

    def _volume_path(self, sample_index: int) -> str:
        return os.path.join(self._processed_root, self._volume_filename(sample_index))

    def _segmentation_path(self, sample_index: int) -> str:
        return os.path.join(self._processed_root, self._segmentation_filename(sample_index))

    def _download_dataset(self) -> None:
        """Downloads the dataset."""
        self._print_license()
        for case_id in tqdm(
            self._get_split_indices(),
            desc=">> Downloading dataset",
            leave=False,
        ):
            image_path, segmentation_path = self._volume_path(case_id), self._segmentation_path(
                case_id
            )
            if os.path.isfile(image_path) and os.path.isfile(segmentation_path):
                continue

            _download_case_with_retry(case_id, image_path, segmentation_path)

    def _preprocess(self):
        """Reorienting the images to LPS and uncompressing them."""

        def _reorient_and_save(path: Path) -> None:
            relative_path = str(path.relative_to(self._root))
            save_path = os.path.join(self._processed_root, relative_path.rstrip(".gz"))
            if os.path.isfile(save_path):
                return
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            nifti.reorient(nib.load(path), "LPS").to_filename(str(save_path))

        compressed_paths = list(Path(self._root).rglob("*.nii.gz"))
        multiprocessing.run_with_threads(
            _reorient_and_save,
            [(path,) for path in compressed_paths],
            num_workers=1,
            progress_desc=">> Preprocessing dataset",
            return_results=False,
        )

        processed_paths = list(Path(self._processed_root).rglob("*.nii"))
        if len(compressed_paths) != len(processed_paths):
            raise RuntimeError(f"Preprocessing failed, missing files in {self._processed_root}.")

    def _print_license(self) -> None:
        """Prints the dataset license."""
        print(f"Dataset license: {self._license}")


def _download_case_with_retry(
    case_id: int,
    image_path: str,
    segmentation_path: str,
    *,
    retries: int = 2,
) -> None:
    for attempt in range(retries):
        try:
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            request.urlretrieve(
                url=f"https://kits19.sfo2.digitaloceanspaces.com/master_{case_id:05d}.nii.gz",  # nosec
                filename=image_path,
            )
            request.urlretrieve(
                url=f"https://raw.githubusercontent.com/neheller/kits23/e282208/dataset/case_{case_id:05d}/segmentation.nii.gz",  # nosec
                filename=segmentation_path,
            )
            return

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5)
            else:
                raise e
