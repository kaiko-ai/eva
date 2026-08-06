"""PathMMU EduContent dataset with text prompts for multimodal VQA."""

import json
import os
import re
from typing import Any, Dict, List, Literal

import huggingface_hub
from loguru import logger
from torchvision import tv_tensors
from torchvision.datasets import utils
from typing_extensions import override

from eva.language.data.messages import MessageSeries, UserMessage
from eva.language.prompts import templates
from eva.language.prompts.templates.preambles import DEFAULT_QA_PREAMBLE
from eva.multimodal.data.datasets.schemas import TransformsSchema
from eva.multimodal.data.datasets.text_image import TextImageDataset
from eva.vision.utils import io


class PathMMUEduContent(TextImageDataset[int]):
    """PathMMU EduContent subset for multimodal VQA evaluation.

    The EduContent subset of PathMMU contains VQA questions derived from
    educational pathology content sourced from OpenPath and Quilt-1M.
    Questions are expert-validated and cover various pathology concepts.

    Both the VQA data and images are available directly from HuggingFace.

    Source:
    - PathMMU: https://huggingface.co/datasets/jamessyx/PathMMU
    """

    _expected_lengths: Dict[str, int] = {
        "val": 146,
        "test": 1683,
        "test_tiny": 255,
    }
    """Expected dataset lengths for each split."""

    _expected_images: int = 3887
    """Expected number of images in images.zip (shared across EduContent and PubMed)."""

    _license: str = "CC-BY-ND-4.0"
    """Dataset license."""

    _default_prompt_template = templates.JsonMultipleChoicePromptTemplate()
    """Default prompt template for formatting questions."""

    _default_render_kwargs: Dict[str, Any] = {
        "preamble": DEFAULT_QA_PREAMBLE,
        "use_option_letters": True,
    }
    """Default kwargs for the template.render() call."""

    def __init__(
        self,
        root: str,
        split: Literal["val", "test", "test_tiny"],
        download: bool = False,
        transforms: TransformsSchema | None = None,
        prompt_template: templates.PromptTemplate | None = None,
        prompt_render_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        """Initialize the PathMMU EduContent dataset.

        Args:
            root: Path to the root directory for storing dataset files.
            split: Dataset split to use. One of "val", "test", or "test_tiny".
            download: Whether to download the dataset if not found locally. When True,
                downloads both images and VQA data from HuggingFace. When False,
                assumes both are available locally and fails if not found.
                Note that the download will be executed only by additionally
                calling the :meth:`prepare_data` method.
            transforms: Transforms to apply to the data samples.
            prompt_template: The template to use for rendering prompts. If None,
                uses the default JSON multiple choice template.
            prompt_render_kwargs: Additional kwargs for rendering the prompt template.
        """
        super().__init__(transforms=transforms)

        self._root = root
        self._split = split
        self._download = download

        self._prompt_template = prompt_template or self._default_prompt_template
        self._prompt_render_kwargs = {
            **self._default_render_kwargs,
            **(prompt_render_kwargs or {}),
        }

        self._samples: List[Dict[str, Any]] = []

    @property
    def classes(self) -> List[str]:
        """Returns the list of answer classes (A-D)."""
        return ["A", "B", "C", "D"]

    @property
    def class_to_idx(self) -> Dict[str, int]:
        """Returns a mapping from class labels to indices."""
        return {label: idx for idx, label in enumerate(self.classes)}

    @property
    def _path_mmu_hf_path(self) -> str:
        """Returns the path to the PathMMU HuggingFace content directory."""
        return os.path.join(self._root, "path_mmu_hf")

    @property
    def _images_path(self) -> str:
        """Returns the path to the extracted images directory."""
        return os.path.join(self._path_mmu_hf_path, "images")

    @override
    def __len__(self) -> int:
        return len(self._samples)

    @override
    def prepare_data(self) -> None:
        """Prepares the dataset by downloading images and VQA data if needed."""
        if self._download:
            self._download_path_mmu_hf()
            self._extract_images()

    @override
    def configure(self) -> None:
        """Configures the dataset by loading VQA samples."""
        self._samples = self._load_vqa_samples()

    @override
    def validate(self) -> None:
        expected_length = self._expected_lengths.get(self._split)
        if expected_length and len(self._samples) != expected_length:
            raise ValueError(
                f"Dataset length mismatch for split '{self._split}': "
                f"expected {expected_length}, but got {len(self._samples)}"
            )

    @override
    def load_text(self, index: int) -> MessageSeries:
        sample = self._samples[index]
        options = sample["options"]

        # Strip letter prefixes from options (e.g., "A) Option text" -> "Option text")
        # since the template will add them back with use_option_letters=True.
        # Only strip if ALL options have uppercase letter prefixes (A-D followed by ) or .)
        all_have_prefix = all(
            opt and len(opt) >= 2 and opt[0].isupper() and opt[0].isalpha() and opt[1] in ")."
            for opt in options
        )
        if all_have_prefix:
            options = [self._strip_letter_prefix(opt) for opt in options]

        prompt = self._prompt_template.render(
            question=sample["question"],
            context=None,
            answer_options=options,
            **self._prompt_render_kwargs,
        )
        return [UserMessage(content=prompt)]

    @staticmethod
    def _strip_letter_prefix(option: str) -> str:
        """Strip letter prefix like 'A) ' or 'A. ' from option text."""
        # Match patterns like "A) ", "A. ", "B) ", etc.
        return re.sub(r"^[A-D][)\.]\s*", "", option)

    @override
    def load_images(self, index: int) -> list[tv_tensors.Image]:
        sample = self._samples[index]
        image_path = os.path.join(self._images_path, sample["img"])
        return [io.read_image_as_tensor(image_path)]

    @override
    def load_target(self, index: int) -> int:
        """Returns the correct answer as an integer index (0=A, 1=B, etc.)."""
        sample = self._samples[index]
        # The answer field contains the full answer starting with the letter
        # e.g., "A) Predominantly multinucleated cells" -> "A" -> 0
        answer_letter = sample["answer"][0]  # Extract just the letter
        if not (answer_letter.isupper() and answer_letter.isalpha()):
            raise ValueError(
                f"Invalid answer format at index {index}: expected uppercase letter (A-D), "
                f"got '{answer_letter}' from answer '{sample['answer']}'"
            )
        return self.class_to_idx[answer_letter]

    @override
    def load_metadata(self, index: int) -> Dict[str, Any]:
        sample = self._samples[index]
        return {
            "answer": sample["answer"],
            "explanation": sample.get("explanation", ""),
            "img": sample["img"],
        }

    def _download_path_mmu_hf(self) -> None:
        """Downloads PathMMU data.json and images.zip from HuggingFace if not already present."""
        data_json_path = os.path.join(self._path_mmu_hf_path, "data.json")
        images_zip_path = os.path.join(self._path_mmu_hf_path, "images.zip")

        if os.path.exists(data_json_path) and os.path.exists(images_zip_path):
            return

        self._print_license()
        logger.info("Downloading PathMMU data from HuggingFace...")
        huggingface_hub.snapshot_download(
            repo_id="jamessyx/PathMMU",
            repo_type="dataset",
            local_dir=self._path_mmu_hf_path,
            allow_patterns=["data.json", "images.zip"],
        )
        logger.info(f"PathMMU HuggingFace data saved to {self._path_mmu_hf_path}")

    def _extract_images(self) -> None:
        """Extracts images from images.zip if not already extracted."""
        if self._are_images_extracted():
            return

        images_zip_path = os.path.join(self._path_mmu_hf_path, "images.zip")
        if not os.path.exists(images_zip_path):
            raise FileNotFoundError(
                f"images.zip not found at '{images_zip_path}'. "
                "Set download=True and call prepare_data() to download the dataset."
            )

        logger.info("Extracting PathMMU images...")
        utils.extract_archive(images_zip_path, self._path_mmu_hf_path, remove_finished=False)
        logger.info(f"Images extracted to {self._images_path}")

    def _are_images_extracted(self) -> bool:
        """Checks if images have already been extracted."""
        if not os.path.isdir(self._images_path):
            return False

        # Count image files in the directory
        image_files = [f for f in os.listdir(self._images_path) if f.endswith((".jpg", ".png"))]
        return len(image_files) >= self._expected_images

    def _load_vqa_samples(self) -> List[Dict[str, Any]]:
        """Loads VQA samples for the current split.

        Returns:
            A list of sample dictionaries with question, options, answer, etc.
        """
        data_json_path = os.path.join(self._path_mmu_hf_path, "data.json")

        if not os.path.exists(data_json_path):
            raise FileNotFoundError(
                f"PathMMU data.json not found at '{data_json_path}'. "
                "Set download=True and call prepare_data() to download the dataset."
            )

        with open(data_json_path, "r") as f:
            data = json.load(f)

        if "EduContent" not in data:
            raise KeyError(
                f"'EduContent' subset not found in data.json. Available keys: {list(data.keys())}"
            )

        educontent_data = data["EduContent"]

        if self._split not in educontent_data:
            raise KeyError(
                f"Split '{self._split}' not found in EduContent subset. "
                f"Available splits: {list(educontent_data.keys())}"
            )

        return educontent_data[self._split]

    def _print_license(self) -> None:
        """Prints the dataset license."""
        logger.info(f"Dataset license: {self._license}")
