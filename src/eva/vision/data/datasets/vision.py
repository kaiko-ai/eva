"""Vision Dataset base class."""

import abc
from typing import Any, Callable, Dict, Generic, List, Tuple, TypeVar

from eva.core.data.datasets import base

InputType = TypeVar("InputType")
"""The input data type."""

TargetType = TypeVar("TargetType")
"""The target data type."""


class VisionDataset(
    base.MapDataset[Tuple[InputType, TargetType, Dict[str, Any]]],
    abc.ABC,
    Generic[InputType, TargetType],
):
    """Base dataset class for vision tasks."""

    def __init__(
        self,
        transforms: Callable | None = None,
    ) -> None:
        """Initializes the dataset.

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

    def __getitem__(self, index: int) -> Tuple[InputType, TargetType, Dict[str, Any]]:
        """Returns the `index`'th data sample.

        Args:
            index: The index of the data sample to load.

        Returns:
            A tuple with the image, the target and the metadata.
        """
        image = self.load_data(index)
        target = self.load_target(index)
        image, target = self._apply_transforms(image, target)
        return image, target, self.load_metadata(index) or {}

    def load_metadata(self, index: int) -> Dict[str, Any] | None:
        """Returns the dataset metadata.

        Args:
            index: The index of the data sample to return the metadata of.

        Returns:
            The sample metadata.
        """

    @abc.abstractmethod
    def load_data(self, index: int) -> InputType:
        """Returns the `index`'th data sample.

        Args:
            index: The index of the data sample to load.

        Returns:
            The sample data.
        """

    @abc.abstractmethod
    def load_target(self, index: int) -> TargetType:
        """Returns the `index`'th target sample.

        Args:
            index: The index of the data sample to load.

        Returns:
            The sample target.
        """

    @abc.abstractmethod
    def filename(self, index: int) -> str:
        """Returns the filename of the `index`'th data sample.

        Note that this is the relative file path to the root.

        Args:
            index: The index of the data-sample to select.

        Returns:
            The filename of the `index`'th data sample.
        """

    def _apply_transforms(
        self, image: InputType, target: TargetType
    ) -> Tuple[InputType, TargetType]:
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
