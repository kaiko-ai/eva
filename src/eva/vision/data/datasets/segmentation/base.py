"""Base for image segmentation datasets."""

import abc
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from torchvision import tv_tensors
from typing_extensions import override

from eva.vision.data.datasets import vision


class ImageSegmentation(vision.VisionDataset[Tuple[tv_tensors.Image, tv_tensors.Mask]], abc.ABC):
    """Image segmentation abstract dataset."""

    def __init__(
        self,
        transforms: Callable | None = None,
    ) -> None:
        """Initializes the image segmentation base class.

        Args:
            transforms: A function/transforms that takes in an
                image and a label and returns the transformed versions of both.
        """
        super().__init__()

        self._transforms = transforms

    @property
    def classes(self) -> List[str] | None:
        """Returns the list with names of the dataset names."""

    @property
    def class_to_idx(self) -> Dict[str, int] | None:
        """Returns a mapping of the class name to its target index."""

    def load_metadata(self, index: int | None) -> Dict[str, Any] | List[Dict[str, Any]] | None:
        """Returns the dataset metadata.

        Args:
            index: The index of the data sample to return the metadata of.
                If `None`, it will return the metadata of the current dataset.

        Returns:
            The sample metadata.
        """

    @abc.abstractmethod
    def load_image(self, index: int) -> tv_tensors.Image:
        """Loads and returns the `index`'th image sample.

        Args:
            index: The index of the data sample to load.

        Returns:
            An image torchvision tensor.
        """

    @abc.abstractmethod
    def load_masks(self, index: int) -> tv_tensors.Mask:
        """Returns the `index`'th target masks sample.

        Args:
            index: The index of the data sample target masks to load.

        Returns:
            The sample masks as a stack of binary torchvision mask
            tensors (label, height, width).
        """

    @abc.abstractmethod
    @override
    def __len__(self) -> int:
        raise NotImplementedError

    @override
    def __getitem__(self, index: int) -> Tuple[tv_tensors.Image, tv_tensors.Mask]:
        image = self.load_image(index)
        masks = self.load_masks(index)
        return self._apply_transforms(image, masks)

    def _apply_transforms(
        self, image: tv_tensors.Image, masks: tv_tensors.Mask
    ) -> Tuple[tv_tensors.Image, tv_tensors.Mask]:
        """Applies the transforms to the provided data and returns them.

        Args:
            image: The desired image.
            masks: The target masks of the image.

        Returns:
            A tuple with the image and the masks transformed.
        """
        if self._transforms is not None:
            image, masks = self._transforms(image, masks)

        return image, masks
