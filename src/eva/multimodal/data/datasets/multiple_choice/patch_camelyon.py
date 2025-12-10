"""PatchCamelyon dataset with text prompts for multimodal classification."""

from typing import Any, Dict, Literal

from torchvision import tv_tensors
from typing_extensions import override

from eva.language.data.messages import MessageSeries, UserMessage
from eva.language.prompts import templates
from eva.language.prompts.templates.preambles import DEFAULT_QA_PREAMBLE
from eva.multimodal.data.datasets.schemas import TransformsSchema
from eva.multimodal.data.datasets.text_image import TextImageDataset
from eva.vision.data import datasets as vision_datasets
from eva.vision.data.datasets import _validators


class PatchCamelyon(TextImageDataset[int], vision_datasets.PatchCamelyon):
    """PatchCamelyon image classification using a multiple choice text prompt."""

    _default_prompt_template = templates.JsonMultipleChoicePromptTemplate()
    """Default prompt template for formatting questions and context."""

    _default_render_kwargs = {
        "preamble": DEFAULT_QA_PREAMBLE,
        "question": "Does this image show metastatic breast tissue?",
        "context": None,
        "answer_options": ["no", "yes"],
        "example_answer": "yes",
    }
    """Default kwargs for the template.render() call."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"],
        download: bool = False,
        transforms: TransformsSchema | None = None,
        max_samples: int | None = None,
        prompt_template: templates.PromptTemplate | None = None,
        prompt_render_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        """Initializes the dataset.

        Args:
            root: The path to the dataset root. This path should contain
                the uncompressed h5 files and the metadata.
            split: The dataset split for training, validation, or testing.
            download: Whether to download the data for the specified split.
                Note that the download will be executed only by additionally
                calling the :meth:`prepare_data` method.
            transforms: Transforms to apply to the data samples.
            max_samples: Maximum number of samples to use. If None, use all samples.
            prompt_template: The template to use for rendering prompts. If None, uses the
                default template which enforces JSON output.
            prompt_render_kwargs: The kwargs to use when rendering the prompt template.
        """
        super().__init__(root=root, split=split, download=download, transforms=transforms)

        self.max_samples = max_samples

        if self.max_samples is not None:
            self._expected_length = {split: max_samples}

        prompt_template = prompt_template or self._default_prompt_template
        prompt_render_kwargs = prompt_render_kwargs or self._default_render_kwargs
        self.prompt = prompt_template.render(**prompt_render_kwargs)

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
    def load_images(self, index: int) -> list[tv_tensors.Image]:
        return [vision_datasets.PatchCamelyon.load_data(self, index)]

    @override
    def load_target(self, index: int) -> int:
        return int(vision_datasets.PatchCamelyon.load_target(self, index).item())

    @override
    def load_metadata(self, index: int) -> Dict[str, Any] | None:
        return vision_datasets.PatchCamelyon.load_metadata(self, index)
