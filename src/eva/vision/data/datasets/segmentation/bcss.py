"""Breast Cancer Semantic Segmentation (BCSS) dataset."""

import os
import time
from typing import Callable, List, Literal

import gdown
from torchvision import tv_tensors
from tqdm import tqdm
from typing_extensions import override

from eva.vision.data.datasets.segmentation import base


class BCSS(base.ImageSegmentation):
    """Breast Cancer Semantic Segmentation (BCSS) dataset."""

    _resource_url = "https://drive.google.com/drive/folders/1zqbdkQF8i5cEmZOGmbdQm-EP8dRYtvss"

    def __init__(
        self,
        root: str,
        split: Literal["train", "test"] | None,
        download: bool = False,
        transforms: Callable | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            root: Path to the root directory of the dataset. The dataset will
                be downloaded and extracted here, if it does not already exist.
            split: Dataset split to use. If `None`, the entire dataset is used.
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
        self._download = download

        self._indices: List[int] = []

    @override
    def prepare_data(self) -> None:
        if self._download:
            self._download_dataset()

    @override
    def filename(self, index: int) -> str:
        return ""

    @override
    def load_image(self, index: int) -> tv_tensors.Image:
        return ""

    @override
    def load_mask(self, index: int) -> tv_tensors.Mask:
        return ""

    @override
    def __len__(self) -> int:
        return len(self._indices)

    def _download_dataset(self) -> None:
        """
        
        google drive limits the download to 50 by using cookies but
        those cookies reset themselves after approximately 40 minutes.
        So i imported time, and added a delay of 60 seconds after 1 picture is downloaded.
        """
        gdrive_files = gdown.download_folder(
            self._resource_url,
            output=self._root,
            quiet=True,
            use_cookies=False,
            remaining_ok=True,
            skip_download=True,
        )
        for file in tqdm(gdrive_files):
            if os.path.isfile(file.local_path):
                continue

            os.makedirs(os.path.dirname(file.local_path), exist_ok=True)
            gdown.download(
                url=f"https://drive.google.com/uc?id={file.id}",
                output=file.local_path,
                quiet=True,
                use_cookies=False,
            )
            time.sleep(60)
