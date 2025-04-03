"""MoNuSAC dataset."""

import functools
import glob
import os
from typing import Any, Callable, Dict, List, Literal
from xml.etree import ElementTree  # nosec

import imagesize
import numpy as np
import numpy.typing as npt
import torch
from skimage import draw
from torchvision import tv_tensors
from torchvision.datasets import utils
from typing_extensions import override

from eva.core.utils.progress_bar import tqdm
from eva.vision.data.datasets import _validators, structs, vision
from eva.vision.utils import io


class MoNuSAC(vision.VisionDataset[tv_tensors.Image, tv_tensors.Mask]):
    """MoNuSAC2020: A Multi-organ Nuclei Segmentation and Classification Challenge.

    Webpage: https://monusac-2020.grand-challenge.org/
    """

    _expected_dataset_lengths: Dict[str, int] = {
        "train": 209,
        "test": 85,
    }
    """Dataset version and split to the expected size."""

    _resources: List[structs.DownloadResource] = [
        structs.DownloadResource(
            filename="MoNuSAC_images_and_annotations.zip",
            url="https://drive.google.com/file/d/1lxMZaAPSpEHLSxGA9KKMt_r-4S8dwLhq/view?usp=sharing",
        ),
        structs.DownloadResource(
            filename="MoNuSAC Testing Data and Annotations.zip",
            url="https://drive.google.com/file/d/1G54vsOdxWY1hG7dzmkeK3r0xz9s-heyQ/view?usp=sharing",
        ),
    ]
    """Resources for the full dataset version."""

    _license: str = (
        "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International "
        "(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)"
    )
    """Dataset license."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "test"],
        export_masks: bool = True,
        download: bool = False,
        transforms: Callable | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            root: Path to the root directory of the dataset. The dataset will
                be downloaded and extracted here, if it does not already exist.
            split: Dataset split to use.
            export_masks: Whether to export, save and use the semantic label masks
                from disk.
            download: Whether to download the data for the specified split.
                Note that the download will be executed only by additionally
                calling the :meth:`prepare_data` method and if the data does not
                exist yet on disk.
            transforms: A function/transforms that takes in an image and a target
                mask and returns the transformed versions of both.
        """
        super().__init__(transforms=transforms)

        self._root = root
        self._split = split
        self._export_masks = export_masks
        self._download = download

    @property
    @override
    def classes(self) -> List[str]:
        return ["Background", "Epithelial", "Lymphocyte", "Neutrophil", "Macrophage", "Ambiguous"]

    @functools.cached_property
    @override
    def class_to_idx(self) -> Dict[str, int]:
        return {label: index for index, label in enumerate(self.classes)}

    @override
    def filename(self, index: int) -> str:
        return os.path.relpath(self._image_files[index], self._root)

    @override
    def prepare_data(self) -> None:
        if self._download:
            self._download_dataset()
        if self._export_masks:
            self._export_semantic_label_masks()

    @override
    def validate(self) -> None:
        _validators.check_dataset_integrity(
            self,
            length=self._expected_dataset_lengths.get(self._split, 0),
            n_classes=6,
            first_and_last_labels=("Background", "Ambiguous"),
        )

    @override
    def load_data(self, index: int) -> tv_tensors.Image:
        image_path = self._image_files[index]
        image_rgb_array = io.read_image(image_path)
        return tv_tensors.Image(image_rgb_array.transpose(2, 0, 1))

    @override
    def load_target(self, index: int) -> tv_tensors.Mask:
        semantic_labels = (
            self._load_semantic_mask_file(index)
            if self._export_masks
            else self._get_semantic_mask(index)
        )
        return tv_tensors.Mask(semantic_labels.squeeze(), dtype=torch.int64)  # type: ignore[reportCallIssue]

    @override
    def __len__(self) -> int:
        return len(self._image_files)

    @functools.cached_property
    def _image_files(self) -> List[str]:
        """Return the list of image files in the dataset.

        Returns:
            List of image file paths.
        """
        files_pattern = os.path.join(self._data_directory, "**", "*.tif")
        image_files = glob.glob(files_pattern, recursive=True)
        return sorted(image_files)

    @functools.cached_property
    def _data_directory(self) -> str:
        """Returns the data directory of the dataset."""
        match self._split:
            case "train":
                directory = "MoNuSAC_images_and_annotations"
            case "test":
                directory = "MoNuSAC Testing Data and Annotations"
            case _:
                raise ValueError(f"Invalid 'split' value '{self._split}'.")

        return os.path.join(self._root, directory)

    def _export_semantic_label_masks(self) -> None:
        """Export semantic label masks to disk."""
        mask_files = [
            (index, filename.replace(".tif", ".npy"))
            for index, filename in enumerate(self._image_files)
        ]
        to_export = filter(lambda x: not os.path.isfile(x[1]), mask_files)
        for sample_index, filename in tqdm(
            list(to_export),
            desc=">> Exporting semantic masks",
            leave=False,
        ):
            semantic_labels = self._get_semantic_mask(sample_index)
            np.save(filename, semantic_labels)

    def _load_semantic_mask_file(self, index: int) -> npt.NDArray[Any]:
        """Load a semantic mask file from disk.

        Args:
            index: Index of the mask file to load.

        Returns:
            Loaded mask as a numpy array.
        """
        mask_filename = self._image_files[index].replace(".tif", ".npy")
        return np.load(mask_filename)

    def _get_semantic_mask(self, index: int) -> npt.NDArray[Any]:
        """Builds and loads the semantic label mask from the XML annotations.

        Args:
            index: Index of the image file.

        Returns:
            Semantic label mask as a numpy array.
        """
        image_path = self._image_files[index]
        width, height = imagesize.get(image_path)
        annotation_path = image_path.replace(".tif", ".xml")
        element_tree = ElementTree.parse(annotation_path)  # nosec
        root = element_tree.getroot()

        semantic_labels = np.zeros((height, width), "uint8")  # type: ignore[reportCallIssue]
        for level in range(len(root)):
            label = [item.attrib["Name"] for item in root[level][0]][0]
            class_id = self.class_to_idx.get(label, self.class_to_idx["Ambiguous"])
            # for the test dataset an additional class 'Ambiguous' was added for
            # difficult regions with fuzzy boundaries
            regions = [item for child in root[level] for item in child if item.tag == "Region"]
            for region in regions:
                vertices = np.array(
                    [(vertex.attrib["X"], vertex.attrib["Y"]) for vertex in region[1]],
                    dtype=np.dtype(float),
                )
                fill_row_coords, fill_col_coords = draw.polygon(
                    vertices[:, 0],
                    vertices[:, 1],
                    (width, height),
                )
                semantic_labels[fill_col_coords, fill_row_coords] = class_id

        return semantic_labels

    def _download_dataset(self) -> None:
        """Downloads the dataset."""
        self._print_license()
        for resource in self._resources:
            if os.path.isdir(self._data_directory):
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
