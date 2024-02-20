"""Base for image classification datasets."""

import abc
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from typing_extensions import override

from eva.vision.data.datasets import vision


class ImageClassification(vision.VisionDataset[Tuple[np.ndarray, np.ndarray]], abc.ABC):
    """Image classification abstract dataset."""

    def __init__(
        self,
        image_transforms: Callable | None = None,
        target_transforms: Callable | None = None,
    ) -> None:
        """Initializes the image classification dataset.

        Args:
            image_transforms: A function/transform that takes in an image
                and returns a transformed version.
            target_transforms: A function/transform that takes in the target
                and transforms it.
        """
        super().__init__()

        self._image_transforms = image_transforms
        self._target_transforms = target_transforms

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
    def load_image(self, index: int) -> np.ndarray:
        """Returns the `index`'th image sample.

        Args:
            index: The index of the data sample to load.

        Returns:
            The image as a numpy array.
        """

    @abc.abstractmethod
    def load_target(self, index: int) -> np.ndarray:
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
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image = self.load_image(index)
        target = self.load_target(index)
        return self._apply_transforms(image, target)

    def _apply_transforms(
        self, image: np.ndarray, target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Applies the transforms to the provided data and returns them.

        Args:
            image: The desired image.
            target: The target of the image.

        Returns:
            A tuple with the image and the target transformed.
        """
        if self._image_transforms is not None:
            image = self._image_transforms(image)

        if self._target_transforms is not None:
            target = self._target_transforms(target)

        return image, target
