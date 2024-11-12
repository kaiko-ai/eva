"""TotalSegmentator 2D segmentation dataset class."""

import functools
import os
from glob import glob
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Tuple

import numpy as np
import numpy.typing as npt
import torch
from torchvision import tv_tensors
from torchvision.datasets import utils
from typing_extensions import override

from eva.core.utils import io as core_io
from eva.core.utils import multiprocessing
from eva.vision.data.datasets import _validators, structs
from eva.vision.data.datasets.segmentation import base
from eva.vision.utils import io


class TotalSegmentator2D(base.ImageSegmentation):
    """TotalSegmentator 2D segmentation dataset."""

    _expected_dataset_lengths: Dict[str, int] = {
        "train_small": 35089,
        "val_small": 1283,
        "train_full": 278190,
        "val_full": 14095,
        "test_full": 25578,
    }
    """Dataset version and split to the expected size."""

    _sample_every_n_slices: int | None = None
    """The amount of slices to sub-sample per 3D CT scan image."""

    _resources_full: List[structs.DownloadResource] = [
        structs.DownloadResource(
            filename="Totalsegmentator_dataset_v201.zip",
            url="https://zenodo.org/records/10047292/files/Totalsegmentator_dataset_v201.zip",
            md5="fe250e5718e0a3b5df4c4ea9d58a62fe",
        ),
    ]
    """Resources for the full dataset version."""

    _resources_small: List[structs.DownloadResource] = [
        structs.DownloadResource(
            filename="Totalsegmentator_dataset_small_v201.zip",
            url="https://zenodo.org/records/10047263/files/Totalsegmentator_dataset_small_v201.zip",
            md5="6b5524af4b15e6ba06ef2d700c0c73e0",
        ),
    ]
    """Resources for the small dataset version."""

    _license: str = (
        "Creative Commons Attribution 4.0 International "
        "(https://creativecommons.org/licenses/by/4.0/deed.en)"
    )
    """Dataset license."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"] | None,
        version: Literal["small", "full"] | None = "full",
        download: bool = False,
        classes: List[str] | None = None,
        optimize_mask_loading: bool = True,
        decompress: bool = True,
        num_workers: int = 10,
        transforms: Callable | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            root: Path to the root directory of the dataset. The dataset will
                be downloaded and extracted here, if it does not already exist.
            split: Dataset split to use. If `None`, the entire dataset is used.
            version: The version of the dataset to initialize. If `None`, it will
                use the files located at root as is and wont perform any checks.
            download: Whether to download the data for the specified split.
                Note that the download will be executed only by additionally
                calling the :meth:`prepare_data` method and if the data does not
                exist yet on disk.
            classes: Whether to configure the dataset with a subset of classes.
                If `None`, it will use all of them.
            optimize_mask_loading: Whether to pre-process the segmentation masks
                in order to optimize the loading time. In the `setup` method, it
                will reformat the binary one-hot masks to a semantic mask and store
                it on disk.
            decompress: Whether to decompress the ct.nii.gz files when preparing the data.
                The label masks won't be decompressed, but when enabling optimize_mask_loading
                it will export the semantic label masks to a single file in uncompressed .nii
                format.
            num_workers: The number of workers to use for optimizing the masks &
                decompressing the .gz files.
            transforms: A function/transforms that takes in an image and a target
                mask and returns the transformed versions of both.

        """
        super().__init__(transforms=transforms)

        self._root = root
        self._split = split
        self._version = version
        self._download = download
        self._classes = classes
        self._optimize_mask_loading = optimize_mask_loading
        self._decompress = decompress
        self._num_workers = num_workers

        if self._optimize_mask_loading and self._classes is not None:
            raise ValueError(
                "To use customize classes please set the optimize_mask_loading to `False`."
            )

        self._samples_dirs: List[str] = []
        self._indices: List[Tuple[int, int]] = []

    @functools.cached_property
    @override
    def classes(self) -> List[str]:
        def get_filename(path: str) -> str:
            """Returns the filename from the full path."""
            return os.path.basename(path).split(".")[0]

        first_sample_labels = os.path.join(
            self._root, self._samples_dirs[0], "segmentations", "*.nii.gz"
        )
        all_classes = sorted(map(get_filename, glob(first_sample_labels)))
        if self._classes:
            is_subset = all(name in all_classes for name in self._classes)
            if not is_subset:
                raise ValueError("Provided class names are not subset of the dataset onces.")

        return all_classes if self._classes is None else self._classes

    @property
    @override
    def class_to_idx(self) -> Dict[str, int]:
        return {label: index for index, label in enumerate(self.classes)}

    @property
    def _file_suffix(self) -> str:
        return "nii" if self._decompress else "nii.gz"

    @override
    def filename(self, index: int) -> str:
        sample_idx, _ = self._indices[index]
        sample_dir = self._samples_dirs[sample_idx]
        return os.path.join(sample_dir, f"ct.{self._file_suffix}")

    @override
    def prepare_data(self) -> None:
        if self._download:
            self._download_dataset()
        if self._decompress:
            self._decompress_files()
        self._samples_dirs = self._fetch_samples_dirs()
        if self._optimize_mask_loading:
            self._export_semantic_label_masks()

    @override
    def configure(self) -> None:
        self._indices = self._create_indices()

    @override
    def validate(self) -> None:
        if self._version is None or self._sample_every_n_slices is not None:
            return

        _validators.check_dataset_integrity(
            self,
            length=self._expected_dataset_lengths.get(f"{self._split}_{self._version}", 0),
            n_classes=len(self._classes) if self._classes else 117,
            first_and_last_labels=(
                (self._classes[0], self._classes[-1])
                if self._classes
                else ("adrenal_gland_left", "vertebrae_T9")
            ),
        )

    @override
    def __len__(self) -> int:
        return len(self._indices)

    @override
    def load_image(self, index: int) -> tv_tensors.Image:
        sample_index, slice_index = self._indices[index]
        image_path = self._get_image_path(sample_index)
        image_array = io.read_nifti(image_path, slice_index)
        image_rgb_array = image_array.repeat(3, axis=2)
        return tv_tensors.Image(image_rgb_array.transpose(2, 0, 1))

    @override
    def load_mask(self, index: int) -> tv_tensors.Mask:
        if self._optimize_mask_loading:
            return self._load_semantic_label_mask(index)
        return self._load_mask(index)

    @override
    def load_metadata(self, index: int) -> Dict[str, Any]:
        _, slice_index = self._indices[index]
        return {"slice_index": slice_index}

    def _load_mask(self, index: int) -> tv_tensors.Mask:
        sample_index, slice_index = self._indices[index]
        semantic_labels = self._load_masks_as_semantic_label(sample_index, slice_index)
        return tv_tensors.Mask(semantic_labels.squeeze(), dtype=torch.int64)  # type: ignore[reportCallIssue]

    def _load_semantic_label_mask(self, index: int) -> tv_tensors.Mask:
        """Loads the segmentation mask from a semantic label NifTi file."""
        sample_index, slice_index = self._indices[index]
        masks_dir = self._get_masks_dir(sample_index)
        filename = os.path.join(masks_dir, "semantic_labels", "masks.nii")
        semantic_labels = io.read_nifti(filename, slice_index)
        return tv_tensors.Mask(semantic_labels.squeeze(), dtype=torch.int64)  # type: ignore[reportCallIssue]

    def _load_masks_as_semantic_label(
        self, sample_index: int, slice_index: int | None = None
    ) -> npt.NDArray[Any]:
        """Loads binary masks as a semantic label mask.

        Args:
            sample_index: The data sample index.
            slice_index: Whether to return only a specific slice.
        """
        masks_dir = self._get_masks_dir(sample_index)
        mask_paths = [os.path.join(masks_dir, f"{label}.nii.gz") for label in self.classes]
        binary_masks = [io.read_nifti(path, slice_index) for path in mask_paths]
        background_mask = np.zeros_like(binary_masks[0])
        return np.argmax([background_mask] + binary_masks, axis=0)

    def _export_semantic_label_masks(self) -> None:
        """Exports the segmentation binary masks (one-hot) to semantic labels."""
        total_samples = len(self._samples_dirs)
        masks_dirs = map(self._get_masks_dir, range(total_samples))
        semantic_labels = [
            (index, os.path.join(directory, "semantic_labels", "masks.nii"))
            for index, directory in enumerate(masks_dirs)
        ]
        to_export = filter(lambda x: not os.path.isfile(x[1]), semantic_labels)

        def _process_mask(sample_index: Any, filename: str) -> None:
            semantic_labels = self._load_masks_as_semantic_label(sample_index)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            io.save_array_as_nifti(semantic_labels, filename)

        multiprocessing.run_with_threads(
            _process_mask,
            list(to_export),
            num_workers=self._num_workers,
            progress_desc=">> Exporting optimized semantic mask",
            return_results=False,
        )

    def _get_image_path(self, sample_index: int) -> str:
        """Returns the corresponding image path."""
        sample_dir = self._samples_dirs[sample_index]
        return os.path.join(self._root, sample_dir, f"ct.{self._file_suffix}")

    def _get_masks_dir(self, sample_index: int) -> str:
        """Returns the directory of the corresponding masks."""
        sample_dir = self._samples_dirs[sample_index]
        return os.path.join(self._root, sample_dir, "segmentations")

    def _get_semantic_labels_filename(self, sample_index: int) -> str:
        """Returns the semantic label filename."""
        masks_dir = self._get_masks_dir(sample_index)
        return os.path.join(masks_dir, "semantic_labels", "masks.nii")

    def _get_number_of_slices_per_sample(self, sample_index: int) -> int:
        """Returns the total amount of slices of a sample."""
        image_path = self._get_image_path(sample_index)
        image_shape = io.fetch_nifti_shape(image_path)
        return image_shape[-1]

    def _fetch_samples_dirs(self) -> List[str]:
        """Returns the name of all the samples of all the splits of the dataset."""
        sample_filenames = [
            filename
            for filename in os.listdir(self._root)
            if os.path.isdir(os.path.join(self._root, filename))
        ]
        return sorted(sample_filenames)

    def _get_split_indices(self) -> List[int]:
        """Returns the samples indices that corresponding the dataset split and version."""
        metadata_file = os.path.join(self._root, "meta.csv")
        metadata = io.read_csv(metadata_file, delimiter=";", encoding="utf-8-sig")

        match self._split:
            case "train":
                image_ids = [item["image_id"] for item in metadata if item["split"] == "train"]
            case "val":
                image_ids = [item["image_id"] for item in metadata if item["split"] == "val"]
            case "test":
                image_ids = [item["image_id"] for item in metadata if item["split"] == "test"]
            case _:
                image_ids = self._samples_dirs

        return sorted(map(self._samples_dirs.index, image_ids))

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
            for slide_idx in range(self._get_number_of_slices_per_sample(sample_idx))
            if slide_idx % (self._sample_every_n_slices or 1) == 0
        ]
        return indices

    def _download_dataset(self) -> None:
        """Downloads the dataset."""
        dataset_resources = {
            "small": self._resources_small,
            "full": self._resources_full,
        }
        resources = dataset_resources.get(self._version or "")
        if resources is None:
            raise ValueError(
                f"Can't download data version '{self._version}'. Use 'small' or 'full'."
            )

        self._print_license()
        for resource in resources:
            if os.path.isdir(self._root):
                continue

            utils.download_and_extract_archive(
                resource.url,
                download_root=self._root,
                filename=resource.filename,
                remove_finished=True,
            )

    def _decompress_files(self) -> None:
        compressed_paths = Path(self._root).rglob("*/ct.nii.gz")
        multiprocessing.run_with_threads(
            core_io.gunzip_file,
            [(str(path),) for path in compressed_paths],
            num_workers=self._num_workers,
            progress_desc=">> Decompressing .gz files",
            return_results=False,
        )

    def _print_license(self) -> None:
        """Prints the dataset license."""
        print(f"Dataset license: {self._license}")
