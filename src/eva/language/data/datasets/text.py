"""Base classes for text-image datasets."""

import abc
from typing import Any, Dict, Generic

from typing_extensions import override

from eva.language.data.datasets.base import LanguageDataset
from eva.language.data.datasets.schemas import TransformsSchema
from eva.language.data.datasets.typings import TargetType, TextSample
from eva.language.data.messages import MessageSeries


class TextDataset(LanguageDataset[TextSample[TargetType]], abc.ABC, Generic[TargetType]):
    """Base dataset class for text-based tasks."""

    def __init__(self, *args, transforms: TransformsSchema | None = None, **kwargs) -> None:
        """Initializes the dataset.

        Args:
            *args: Positional arguments for the base class.
            transforms: The transforms to apply to the text and target when
                loading the samples.
            **kwargs: Keyword arguments for the base class.
        """
        super().__init__(*args, **kwargs)

        self.transforms = transforms

    def load_metadata(self, index: int) -> Dict[str, Any] | None:
        """Returns the dataset metadata.

        Args:
            index: The index of the data sample.

        Returns:
            The sample metadata.
        """

    @abc.abstractmethod
    def load_text(self, index: int) -> MessageSeries:
        """Returns the text content.

        Args:
            index: The index of the data sample.

        Returns:
            The text content.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_target(self, index: int) -> TargetType:
        """Returns the target label.

        Args:
            index: The index of the data sample.

        Returns:
            The target label.
        """
        raise NotImplementedError

    @override
    def __getitem__(self, index: int) -> TextSample[TargetType]:
        item = TextSample(
            text=self.load_text(index),
            target=self.load_target(index),
            metadata=self.load_metadata(index) or {},
        )
        return self._apply_transforms(item)

    def _apply_transforms(self, sample: TextSample[TargetType]) -> TextSample[TargetType]:
        """Applies the dataset transforms to the text and target.

        Args:
            sample: The text sample..

        Returns:
            The transformed sample.
        """
        if self.transforms:
            text = self.transforms.text(sample.text) if self.transforms.text else sample.text
            target = (
                self.transforms.target(sample.target) if self.transforms.target else sample.target
            )
            return TextSample(
                text=text,
                target=target,
                metadata=sample.metadata,
            )
        return sample
