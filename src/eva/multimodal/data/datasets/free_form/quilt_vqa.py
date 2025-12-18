"""QuiltVQA dataset class."""

import os
from typing import Any, Dict, Literal

from datasets import Dataset, load_dataset, load_from_disk
from loguru import logger
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from typing_extensions import override

from eva.language.data.messages import MessageSeries, UserMessage
from eva.language.prompts import templates
from eva.multimodal.data.datasets.schemas import TransformsSchema
from eva.multimodal.data.datasets.text_image import TextImageDataset
from eva.multimodal.prompts.templates.preambles import DEFAULT_VQA_PREAMBLE


class QuiltVQA(TextImageDataset[str]):
    """Dataset class for Quilt_VQA.

    Source: https://huggingface.co/datasets/wisdomik/Quilt_VQA
    """

    _expected_dataset_lengths: Dict[str | None, int] = {"test": 985, None: 985}
    """Expected dataset lengths for the splits and complete dataset."""

    _license: str = "CC-BY-NC-ND-3.0 (https://creativecommons.org/licenses/by-nc-nd/3.0/ch/deed.de)"
    """Dataset license."""

    _default_prompt_template: templates.PromptTemplate = (
        templates.RawFreeFormQuestionPromptTemplate()
    )
    """Default prompt template for formatting questions and context."""

    _default_render_kwargs: Dict[str, Any] = {
        "preamble": DEFAULT_VQA_PREAMBLE,
    }
    """Default kwargs for the template.render() call."""

    def __init__(
        self,
        root: str | None = None,
        split: Literal["test"] | None = None,
        download: bool = False,
        transforms: TransformsSchema | None = None,
        max_samples: int | None = None,
        prompt_template: templates.PromptTemplate | None = None,
        prompt_render_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        """Initialize the QuiltVQA dataset.

        Args:
            root: Directory to cache the dataset. If None, no local caching is used.
            split: Valid splits among ["test"]. If None, uses the entire dataset.
            download: Whether to download the dataset if not found locally. Default is False.
            transforms: Transforms to apply to the data samples.
            max_samples: Maximum number of samples to use. If None, use all samples.
            prompt_template: The template to use for rendering prompts. If None, uses the
                default template which enforces JSON output.
            prompt_render_kwargs: The kwargs to use when rendering the prompt template.
        """
        super().__init__(transforms=transforms)

        self._root = root
        self._split = split
        self._download = download
        self._max_samples = max_samples

        self.prompt_template = prompt_template or self._default_prompt_template
        self.prompt_render_kwargs = prompt_render_kwargs or self._default_render_kwargs

        self.dataset: Dataset

    @override
    def __len__(self) -> int:
        return len(self.dataset)

    @override
    def prepare_data(self) -> None:
        """Downloads and prepares the QuiltVQA dataset.

        If `self._root` is None, the dataset is used directly from HuggingFace.
        Otherwise, it checks if the dataset is already cached in `self._root`.
        If not cached, it downloads the dataset into `self._root`.
        """
        dataset_path = None
        if self._root:
            if os.path.exists(os.path.join(self._root, "test")):
                dataset_path = os.path.join(self._root, "test")
            else:
                dataset_path = self._root

        self.dataset = self._load_dataset(dataset_path)

    @override
    def validate(self) -> None:
        if self._split not in ["test", None]:
            raise ValueError(f"Available splits are ['test', None], but got '{self._split}'")
        if len(self) != (self._max_samples or self._expected_dataset_lengths[self._split]):
            raise ValueError(
                f"Dataset length mismatch for split '{self._split}': "
                f"expected {self._expected_dataset_lengths[self._split]}, "
                f"but got {len(self)}"
            )

    @override
    def load_text(self, index: int) -> MessageSeries:
        if index < 0 or index >= len(self.dataset):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.dataset)}")
        sample = dict(self.dataset[index])
        prompt = self.prompt_template.render(
            question=sample["question"],
            **self.prompt_render_kwargs,
        )
        return [UserMessage(content=prompt)]

    @override
    def load_images(self, index: int) -> list[tv_tensors.Image]:
        return [F.to_image(self.dataset[index]["image"])]

    @override
    def load_target(self, index: int) -> str:
        if index < 0 or index >= len(self.dataset):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.dataset)}")
        return self.dataset[index]["answer"]

    @override
    def load_metadata(self, index: int) -> Dict[str, str]:
        sample = self.dataset[index]
        return {
            "answer_type": sample["answer_type"],
            "context": sample["context"],
        }

    def _print_license(self) -> None:
        """Prints the dataset license."""
        print(f"Dataset license: {self._license}")

    def _load_dataset(self, dataset_path: str | None) -> Dataset:
        """Loads the QuiltVQA dataset from the local cache or downloads it.

        Args:
            dataset_path: The path to the local cache (may be None).

        Returns:
            The loaded dataset object.
        """
        dataset_name = "wisdomik/Quilt_VQA"

        if self._download:
            logger.info("Downloading dataset from HuggingFace Hub")
            raw_dataset = load_dataset(
                dataset_name,
                split="train",
                # labelled as "train" but this loads the test file quiltvqa_test_w_ans.json
                trust_remote_code=True,
                download_mode="reuse_dataset_if_exists",
            )
            if dataset_path:
                os.makedirs(dataset_path, exist_ok=True)
                raw_dataset.save_to_disk(dataset_path)  # type: ignore
                logger.info(f"Dataset saved to: {dataset_path}")
        else:
            if not dataset_path or not os.path.exists(dataset_path):
                raise ValueError(
                    "Dataset path not found. Set download=True or provide a valid root path."
                )

            logger.info(f"Loading dataset from: {dataset_path}")
            raw_dataset = load_from_disk(dataset_path)

        return raw_dataset  # type: ignore
