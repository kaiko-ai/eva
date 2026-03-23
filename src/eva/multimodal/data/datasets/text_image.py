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
    def load_images(self, index: int) -> list[tv_tensors.Image]:
        """Returns the images for a sample.

        Args:
            index: The index of the data sample.

        Returns:
            A list of image tensors.
        """
        raise NotImplementedError

    @override
    def __getitem__(self, index: int) -> TextImageSample[TargetType]:
        item = TextImageSample(
            text=self.load_text(index),
            images=self.load_images(index),
            target=self.load_target(index),
            metadata=self.load_metadata(index) or {},
        )
        return self._apply_transforms(item)

    @override
    def _apply_transforms(self, sample: TextImageSample[TargetType]) -> TextImageSample[TargetType]:
        """Applies the dataset transforms to the text, images and target.

        Args:
            sample: The sample containing text, images, target and metadata.

        Returns:
            The transformed sample.
        """
        if self.transforms:
            text = self.transforms.text(sample.text) if self.transforms.text else sample.text
            if self.transforms.image:
                images = [self.transforms.image(img) for img in sample.images]
            else:
                images = sample.images
            target = (
                self.transforms.target(sample.target) if self.transforms.target else sample.target
            )
            return TextImageSample(
                text=text,
                images=images,
                target=target,
                metadata=sample.metadata,
            )
        return sample
