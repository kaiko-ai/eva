"""Base for image classification datasets."""

import abc
from typing import Any, Callable, Dict, List, Tuple

import torch
from torchvision import tv_tensors
from typing_extensions import override

from eva.vision.data.datasets import vision


class ImageClassification(vision.VisionDataset[Tuple[tv_tensors.Image, torch.Tensor]], abc.ABC):
    """Image classification abstract dataset."""

    def __init__(
        self,
        transforms: Callable | None = None,
    ) -> None:
        """Initializes the image classification dataset.

        Args:
            transforms: A function/transform which returns a transformed
                version of the raw data samples.
        """
        super().__init__()

        self._transforms = transforms

    @property
    def classes(self) -> List[str] | None:
        """Returns the list with names of the dataset names."""

    @property
    def class_to_idx(self) -> Dict[str, int] | None:
        """Returns a mapping of the class name to its target index."""

    def load_metadata(self, index: int) -> Dict[str, Any] | None:
        """Returns the dataset metadata.

        Args:
            index: The index of the data sample to return the metadata of.

        Returns:
            The sample metadata.
        """

    @abc.abstractmethod
    def load_image(self, index: int) -> tv_tensors.Image:
        """Returns the `index`'th image sample.

        Args:
            index: The index of the data sample to load.

        Returns:
            The image as a numpy array.
        """

    @abc.abstractmethod
    def load_target(self, index: int) -> torch.Tensor:
        """Returns the `index`'th target sample.

        Args:
            index: The index of the data sample to load.

        Returns:
            The sample target as an array.
        """

    @abc.abstractmethod
    @override
    def __len__(self) -> int:
        raise NotImplementedError

    @override
    def __getitem__(self, index: int) -> Tuple[tv_tensors.Image, torch.Tensor, Dict[str, Any]]:
        image = self.load_image(index)
        target = self.load_target(index)
        image, target = self._apply_transforms(image, target)
        return image, target, self.load_metadata(index) or {}

    def _apply_transforms(
        self, image: tv_tensors.Image, target: torch.Tensor
    ) -> Tuple[tv_tensors.Image, torch.Tensor]:
        """Applies the transforms to the provided data and returns them.

        Args:
            image: The desired image.
            target: The target of the image.

        Returns:
            A tuple with the image and the target transformed.
        """
        if self._transforms is not None:
            image, target = self._transforms(image, target)
        return image, target
