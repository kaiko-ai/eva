"""Base for image segmentation datasets."""

import abc
from typing import Any, Callable, Dict, List, Tuple

from torchvision import tv_tensors
from typing_extensions import override

from eva.vision.data.datasets import vision


class ImageSegmentation(vision.VisionDataset[Tuple[tv_tensors.Image, tv_tensors.Mask]], abc.ABC):
    """Image segmentation abstract dataset."""

    def __init__(self, transforms: Callable | None = None) -> None:
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

    @abc.abstractmethod
    def load_image(self, index: int) -> tv_tensors.Image:
        """Loads and returns the `index`'th image sample.

        Args:
            index: The index of the data sample to load.

        Returns:
            An image torchvision tensor (channels, height, width).
        """

    @abc.abstractmethod
    def load_mask(self, index: int) -> tv_tensors.Mask:
        """Returns the `index`'th target masks sample.

        Args:
            index: The index of the data sample target masks to load.

        Returns:
            The semantic mask as a (H x W) shaped tensor with integer
            values which represent the pixel class id.
        """

    def load_metadata(self, index: int) -> Dict[str, Any] | None:
        """Returns the dataset metadata.

        Args:
            index: The index of the data sample to return the metadata of.
                If `None`, it will return the metadata of the current dataset.

        Returns:
            The sample metadata.
        """

    @abc.abstractmethod
    @override
    def __len__(self) -> int:
        raise NotImplementedError

    @override
    def __getitem__(self, index: int) -> Tuple[tv_tensors.Image, tv_tensors.Mask, Dict[str, Any]]:
        image = self.load_image(index)
        mask = self.load_mask(index)
        metadata = self.load_metadata(index) or {}
        image_tensor, mask_tensor = self._apply_transforms(image, mask)
        return image_tensor, mask_tensor, metadata

    def _apply_transforms(
        self, image: tv_tensors.Image, mask: tv_tensors.Mask
    ) -> Tuple[tv_tensors.Image, tv_tensors.Mask]:
        """Applies the transforms to the provided data and returns them.

        Args:
            image: The desired image.
            mask: The target segmentation mask.

        Returns:
            A tuple with the image and the masks transformed.
        """
        if self._transforms is not None:
            image, mask = self._transforms(image, mask)

        return image, mask
