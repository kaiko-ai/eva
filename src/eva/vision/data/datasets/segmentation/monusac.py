"""MoNuSAC dataset."""

import functools
import glob
import os
from typing import Any, Callable, Dict, List
from xml.etree import ElementTree  # nosec

import imagesize
import numpy as np
import numpy.typing as npt
import torch
from skimage import draw
from torchvision import tv_tensors
from torchvision.datasets import utils
from typing_extensions import override

from eva.vision.data.datasets import structs
from eva.vision.data.datasets.segmentation import base
from eva.vision.utils import io


class MoNuSAC(base.ImageSegmentation):
    """MoNuSAC2020: A Multi-organ Nuclei Segmentation and Classification Challenge.

    Webpage: https://monusac-2020.grand-challenge.org/
    """

    _resources: List[structs.DownloadResource] = [
        structs.DownloadResource(
            filename="MoNuSAC_images_and_annotations.zip",
            url="https://drive.google.com/file/d/1lxMZaAPSpEHLSxGA9KKMt_r-4S8dwLhq/view?usp=sharing",
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
        download: bool = False,
        transforms: Callable | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            root: Path to the root directory of the dataset. The dataset will
                be downloaded and extracted here, if it does not already exist.
            download: Whether to download the data for the specified split.
                Note that the download will be executed only by additionally
                calling the :meth:`prepare_data` method and if the data does not
                exist yet on disk.
            transforms: A function/transforms that takes in an image and a target
                mask and returns the transformed versions of both.
        """
        super().__init__(transforms=transforms)

        self._root = root
        self._download = download

    @property
    @override
    def classes(self) -> List[str]:
        return ["Epithelial", "Lymphocyte", "Neutrophil", "Macrophage"]

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

    @override
    def load_image(self, index: int) -> tv_tensors.Image:
        image_path = self._image_files[index]
        image_rgb_array = io.read_image(image_path)
        return tv_tensors.Image(image_rgb_array.transpose(2, 0, 1))

    @override
    def load_mask(self, index: int) -> tv_tensors.Mask:
        semantic_labels = self._export_semantic_label_mask(index)
        return tv_tensors.Mask(semantic_labels.squeeze(), dtype=torch.int64)

    @override
    def __len__(self) -> int:
        return len(self._image_files)

    @functools.cached_property
    def _image_files(self) -> List[str]:
        files_pattern = os.path.join(self._root, "MoNuSAC_images_and_annotations", "**", "*.tif")
        image_files = glob.glob(files_pattern, recursive=True)
        return sorted(image_files)

    def _export_semantic_label_mask(self, index) -> npt.NDArray[Any]:
        image_path = self._image_files[index]
        image_size = imagesize.get(image_path)
        annotation_path = image_path.replace(".tif", ".xml")
        element_tree = ElementTree.parse(annotation_path)  # nosec
        root = element_tree.getroot()

        semantic_labels = np.zeros(image_size, "uint8")
        for level in range(len(root)):
            label = [item.attrib["Name"] for item in root[level][0]][0]
            regions = [item for child in root[level] for item in child if item.tag == "Region"]
            for region in regions:
                vertices = np.array(
                    [(vertex.attrib["X"], vertex.attrib["Y"]) for vertex in region[1]],
                    dtype=np.dtype(int),
                )
                fill_row_coords, fill_col_coords = draw.polygon(
                    vertices[:, 0],
                    vertices[:, 1],
                    image_size,
                )
                semantic_labels[fill_row_coords, fill_col_coords] = self.class_to_idx[label] + 1


            # for child in root[level]:
            #     for item in child:
            #         if item.tag != "Region":
            #             continue

            #         vertices = np.array(
            #             [(vertex.attrib["X"], vertex.attrib["Y"]) for vertex in item[1]],
            #             dtype=np.dtype(int),
            #         )
            #         fill_row_coords, fill_col_coords = draw.polygon(
            #             vertices[:, 0],
            #             vertices[:, 1],
            #             image_size,
            #         )
            #         semantic_labels[fill_row_coords, fill_col_coords] = self.class_to_idx[label] + 1

        return semantic_labels

    def _download_dataset(self) -> None:
        """Downloads the dataset."""
        self._print_license()
        for resource in self._resources:
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
