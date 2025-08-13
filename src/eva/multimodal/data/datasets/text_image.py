"""Base classes for text-image datasets."""

import abc
from typing import Generic

from torchvision import tv_tensors
from typing_extensions import override

from eva.language.data.datasets.text import TextDataset
from eva.multimodal.data.datasets.base import MultimodalDataset
from eva.multimodal.data.datasets.schemas import TransformsSchema
from eva.multimodal.data.datasets.typings import TargetType, TextImageSample


class TextImageDataset(
    MultimodalDataset[TextImageSample[TargetType]], TextDataset, abc.ABC, Generic[TargetType]
):
    """Base dataset class for text-image tasks."""

    def __init__(self, *args, transforms: TransformsSchema | None = None, **kwargs) -> None:
        """Initializes the dataset.

        Args:
            *args: Positional arguments for the base class.
            transforms: The transforms to apply to the text, image and target when
                loading the samples.
            **kwargs: Keyword arguments for the base class.
        """
        super().__init__(*args, **kwargs)

        self.transforms = transforms

    @abc.abstractmethod
    def load_image(self, index: int) -> tv_tensors.Image:
        """Returns the image content.

        Args:
            index: The index of the data sample.

        Returns:
            The image content.
        """
        raise NotImplementedError

    @override
    def __getitem__(self, index: int) -> TextImageSample[TargetType]:
        item = TextImageSample(
            text=self.load_text(index),
            image=self.load_image(index),
            target=self.load_target(index),
            metadata=self.load_metadata(index) or {},
        )
        return self._apply_transforms(item)

    @override
    def _apply_transforms(self, sample: TextImageSample[TargetType]) -> TextImageSample[TargetType]:
        """Applies the dataset transforms to the text, image and target.

        Args:
            sample: The sample containing text, image, target and metadata.

        Returns:
            The transformed sample.
        """
        if self.transforms:
            text = self.transforms.text(sample.text) if self.transforms.text else sample.text
            image = self.transforms.image(sample.image) if self.transforms.image else sample.image
            target = (
                self.transforms.target(sample.target) if self.transforms.target else sample.target
            )
            return TextImageSample(
                text=text,
                image=image,
                target=target,
                metadata=sample.metadata,
            )
        return sample
