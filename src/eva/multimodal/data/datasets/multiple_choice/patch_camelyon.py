"""PatchCamelyon dataset with text prompts for multimodal classification."""

from typing import Any, Dict, Literal

from torchvision import tv_tensors
from typing_extensions import override

from eva.language.data.messages import MessageSeries, UserMessage
from eva.language.prompts.templates.json import JsonMultipleChoicePromptTemplate
from eva.multimodal.data.datasets.schemas import TransformsSchema
from eva.multimodal.data.datasets.text_image import TextImageDataset
from eva.vision.data import datasets as vision_datasets
from eva.vision.data.datasets import _validators


class PatchCamelyon(TextImageDataset[int], vision_datasets.PatchCamelyon):
    """PatchCamelyon image classification using a multiple choice text prompt."""

    _prompt_template = JsonMultipleChoicePromptTemplate()
    _question: str = "Does this image show metastatic breast tissue?"

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"],
        download: bool = False,
        transforms: TransformsSchema | None = None,
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
            max_samples: Maximum number of samples to use. If None, use all samples.
        """
        super().__init__(root=root, split=split, download=download, transforms=transforms)

        self.max_samples = max_samples
        self.prompt = self._render_prompt()

        if self.max_samples is not None:
            self._expected_length = {split: max_samples}

    def _render_prompt(self) -> str:
        return self._prompt_template.render(
            question=self._question,
            context=None,
            answer_options=self.classes,
            example_answer=self.classes[0],
            example_reason="Key visual evidence from the histopathology image.",
        )

    @property
    @override
    def classes(self) -> list[str]:
        return ["no", "yes"]

    @property
    @override
    def class_to_idx(self) -> Dict[str, int]:
        return {label: idx for idx, label in enumerate(self.classes)}

    @override
    def __len__(self) -> int:
        return self.max_samples or self._fetch_dataset_length()

    @override
    def validate(self) -> None:
        _validators.check_dataset_integrity(
            self,
            length=self._expected_length.get(self._split, 0),
            n_classes=2,
            first_and_last_labels=("no", "yes"),
        )

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
