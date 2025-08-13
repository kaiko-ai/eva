"""PatchCamelyon dataset with text prompts for multimodal classification."""

from typing import Any, Dict, Literal

from torchvision import tv_tensors
from typing_extensions import override

from eva.language.data.messages import MessageSeries, UserMessage
from eva.multimodal.data.datasets.schemas import TransformsSchema
from eva.multimodal.data.datasets.text_image import TextImageDataset
from eva.vision.data import datasets as vision_datasets


class PatchCamelyon(TextImageDataset[int], vision_datasets.PatchCamelyon):
    """PatchCamelyon image classification using a multiple choice text prompt."""

    _default_prompt = (
        "You are a pathology expert helping pathologists to analyze images of tissue samples.\n"
        "Question: Does this image show metastatic breast tissue?\n"
        "Options: A: no, B: yes\n"
        "Only answer with a single letter without further explanation. "
        "Please always provide an answer, even if you are not sure.\n"
        "Answer: "
    )

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"],
        download: bool = False,
        transforms: TransformsSchema | None = None,
        prompt: str | None = None,
        max_samples: int | None = None,
    ) -> None:
        """Initializes the dataset.

        Args:
            root: The path to the dataset root. This path should contain
                the uncompressed h5 files and the metadata.
            split: The dataset split for training, validation, or testing.
            download: Whether to download the data for the specified split.
                Note that the download will be executed only by additionally
                calling the :meth:`prepare_data` method.
            transforms: A function/transform which returns a transformed
                version of the raw data samples.
            prompt: The text prompt to use for classification (multple choice).
            max_samples: Maximum number of samples to use. If None, use all samples.
        """
        super().__init__(root=root, split=split, download=download, transforms=transforms)

        self.max_samples = max_samples
        self.prompt = prompt or self._default_prompt

        if self.max_samples is not None:
            self._expected_length = {split: max_samples}

    @property
    @override
    def class_to_idx(self) -> Dict[str, int]:
        return {"A": 0, "B": 1}

    @override
    def __len__(self) -> int:
        return self.max_samples or self._fetch_dataset_length()

    @override
    def load_text(self, index: int) -> MessageSeries:
        return [UserMessage(content=self.prompt)]

    @override
    def load_image(self, index: int) -> tv_tensors.Image:
        return vision_datasets.PatchCamelyon.load_data(self, index)

    @override
    def load_target(self, index: int) -> int:
        return int(vision_datasets.PatchCamelyon.load_target(self, index).item())

    @override
    def load_metadata(self, index: int) -> Dict[str, Any] | None:
        return vision_datasets.PatchCamelyon.load_metadata(self, index)
