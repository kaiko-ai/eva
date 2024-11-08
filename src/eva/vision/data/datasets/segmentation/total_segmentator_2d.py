"""TotalSegmentator 2D segmentation dataset class."""

import functools
import os
from glob import glob
from typing import Any, Callable, Dict, List, Literal, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torchvision import tv_tensors
from torchvision.datasets import utils
from typing_extensions import override

from eva.core.utils.progress_bar import tqdm
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
            decompress: Whether to decompress the .gz files when preparing the data.
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
    def file_suffix(self) -> str:
        return ".nii" if self._decompress else ".nii.gz"

    @override
    def filename(self, index: int, segmented: bool = True) -> str:
        sample_idx, _ = self._indices[index]
        sample_dir = self._samples_dirs[sample_idx]
        return os.path.join(sample_dir, "ct.nii.gz")

    @override
    def prepare_data(self) -> None:
        if self._download:
            self._download_dataset()
        self._samples_dirs = self._fetch_samples_dirs()
        if self._optimize_mask_loading:
            self._export_semantic_label_masks()
        if self._decompress:
            compressed_paths = Path(self._root).rglob("*/ct.nii.gz")
            with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
                futures = {
                    executor.submit(gunzip_file, str(path)): path
                    for path in compressed_paths
                }
                with tqdm(total=len(futures), desc=">> Decompressing .gz files", leave=False) as pbar:
                    for future in as_completed(futures):
                        filename = futures[future]
                        try:
                            future.result()
                            pbar.update(1)
                        except Exception as e:
                            print(f"Error processing {filename}: {str(e)}")

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

        if self._decompress:
            # TODO: maybe use file_suffix property instead as before
            image_path = self._uncompressed_path(image_path)
            
        image_array = io.read_nifti(image_path, slice_index)
        image_rgb_array = image_array.repeat(3, axis=2)
        return tv_tensors.Image(image_rgb_array.transpose(2, 0, 1))

    @override
    def load_mask(self, index: int) -> tv_tensors.Mask:
        return torch.tensor(1)
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
        filename = os.path.join(masks_dir, "semantic_labels", "masks.nii.gz")
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
        mask_paths = [os.path.join(masks_dir, label + ".nii.gz") for label in self.classes]
        binary_masks = [io.read_nifti(path, slice_index) for path in mask_paths]
        background_mask = np.zeros_like(binary_masks[0])
        return np.argmax([background_mask] + binary_masks, axis=0)

    def _export_semantic_label_masks(self) -> None:
        """Exports the segmentation binary masks (one-hot) to semantic labels."""
        total_samples = len(self._samples_dirs)
        masks_dirs = map(self._get_masks_dir, range(total_samples))
        semantic_labels = [
            (index, os.path.join(directory, "semantic_labels", "masks.nii.gz"))
            for index, directory in enumerate(masks_dirs)
        ]
        to_export = filter(lambda x: not os.path.isfile(x[1]), semantic_labels)

        def _process_mask(sample_index: Any, filename: str) -> None:
            """Process a single sample with error handling"""
            semantic_labels = self._load_masks_as_semantic_label(sample_index)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            io.save_array_as_nifti(semantic_labels, filename)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(_process_mask, sample_index, filename): filename 
                for sample_index, filename in to_export
            }
            
            with tqdm(total=len(futures), desc=">> Exporting optimized semantic masks", leave=False) as pbar:
                for future in as_completed(futures):
                    filename = futures[future]
                    try:
                        future.result()
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")

    def _get_image_path(self, sample_index: int) -> str:
        """Returns the corresponding image path."""
        sample_dir = self._samples_dirs[sample_index]
        return os.path.join(self._root, sample_dir, "ct.nii.gz")

    def _get_masks_dir(self, sample_index: int) -> str:
        """Returns the directory of the corresponding masks."""
        sample_dir = self._samples_dirs[sample_index]
        return os.path.join(self._root, sample_dir, "segmentations")
        
    def _uncompressed_path(self, path: str) -> str:
        """Returns the uncompressed path of a file."""
        return os.path.join(os.path.dirname(path), os.path.basename(path).replace(".nii.gz", ".nii"))

    def _get_semantic_labels_filename(self, sample_index: int) -> str:
        """Returns the semantic label filename."""
        masks_dir = self._get_masks_dir(sample_index)
        return os.path.join(masks_dir, "semantic_labels", "masks.nii.gz")

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

    def _print_license(self) -> None:
        """Prints the dataset license."""
        print(f"Dataset license: {self._license}")


def gunzip_file(path: str, unpack_dir: str | None = None) -> str:
    """Unpacks a .gz file to the provided directory.

    Args:
        path: Path to the .gz file to extract.
        unpack_dir: Directory to extract the file to. If `None`, it will use the
            same directory as the compressed file.

    Returns:
        The path to the extracted file.
    """
    unpack_dir = unpack_dir or os.path.dirname(path)
    save_path = os.path.join(unpack_dir, os.path.basename(path).replace(".gz", ""))
    if not os.path.isfile(save_path):
        with gzip.open(path, "rb") as f_in:
            with open(save_path, "wb") as f_out:
                f_out.write(f_in.read())
    return save_path

_grouped_class_mappings = {
    # Abdominal Organs
    'spleen': 'spleen',
    'kidney_right': 'kidney',
    'kidney_left': 'kidney',
    'gallbladder': 'gallbladder',
    'liver': 'liver',
    'stomach': 'stomach',
    'pancreas': 'pancreas',
    'small_bowel': 'small_bowel',
    'duodenum': 'duodenum',
    'colon': 'colon',
    
    # Endocrine System
    'adrenal_gland_right': 'adrenal_gland',
    'adrenal_gland_left': 'adrenal_gland',
    'thyroid_gland': 'thyroid_gland',
    
    # Respiratory System
    'lung_upper_lobe_left': 'lungs',
    'lung_lower_lobe_left': 'lungs',
    'lung_upper_lobe_right': 'lungs',
    'lung_middle_lobe_right': 'lungs',
    'lung_lower_lobe_right': 'lungs',
    'trachea': 'trachea',
    'esophagus': 'esophagus',
    
    # Urogenital System
    'urinary_bladder': 'urogenital_system',
    'prostate': 'urogenital_system',
    'kidney_cyst_left': 'kidney_cyst',
    'kidney_cyst_right': 'kidney_cyst',
    
    # Vertebral Column
    **{f'vertebrae_{v}': 'vertebrae' for v in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']},
    **{f'vertebrae_{v}': 'vertebrae' for v in [f'T{i}' for i in range(1, 13)]},
    **{f'vertebrae_{v}': 'vertebrae' for v in [f'L{i}' for i in range(1, 6)]},
    'vertebrae_S1': 'vertebrae',
    'sacrum': 'sacral_spine',
    
    # Cardiovascular System
    'heart': 'heart',
    'aorta': 'arteries',
    'pulmonary_vein': 'veins',
    'brachiocephalic_trunk': 'arteries',
    'subclavian_artery_right': 'arteries',
    'subclavian_artery_left': 'arteries',
    'common_carotid_artery_right': 'arteries',
    'common_carotid_artery_left': 'arteries',
    'brachiocephalic_vein_left': 'veins',
    'brachiocephalic_vein_right': 'veins',
    'atrial_appendage_left': 'atrial_appendage_left',
    'superior_vena_cava': 'veins',
    'inferior_vena_cava': 'veins',
    'portal_vein_and_splenic_vein': 'veins',
    'iliac_artery_left': 'arteries',
    'iliac_artery_right': 'arteries',
    'iliac_vena_left': 'veins',
    'iliac_vena_right': 'veins',
    
    # Upper Extremity Bones
    'humerus_left': 'humerus',
    'humerus_right': 'humerus',
    'scapula_left': 'scapula',
    'scapula_right': 'scapula',
    'clavicula_left': 'clavicula',
    'clavicula_right': 'clavicula',
    
    # Lower Extremity Bones
    'femur_left': 'femur',
    'femur_right': 'femur',
    'hip_left': 'hip',
    'hip_right': 'hip',
    
    # Muscles
    'gluteus_maximus_left': 'gluteus',
    'gluteus_maximus_right': 'gluteus',
    'gluteus_medius_left': 'gluteus',
    'gluteus_medius_right': 'gluteus',
    'gluteus_minimus_left': 'gluteus',
    'gluteus_minimus_right': 'gluteus',
    'autochthon_left': 'autochthon',
    'autochthon_right': 'autochthon',
    'iliopsoas_left': 'iliopsoas',
    'iliopsoas_right': 'iliopsoas',
    
    # Central Nervous System
    'brain': 'brain',
    'spinal_cord': 'spinal_cord',
    
    # Skull and Thoracic Cage
    'skull': 'skull',
    **{f'rib_left_{i}': 'ribs' for i in range(1, 13)},
    **{f'rib_right_{i}': 'ribs' for i in range(1, 13)},
    'costal_cartilages': 'ribs',
    'sternum': 'sternum',
}