"""PathMMU Atlas dataset with text prompts for multimodal VQA."""

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


class PathMMUAtlas(TextImageDataset[int]):
    """PathMMU Atlas subset for multimodal VQA evaluation.

    The Atlas subset of PathMMU contains VQA questions derived from the
    ARCH Book Set pathology images. Questions are expert-validated and
    cover various pathology concepts from educational textbooks.

    This dataset combines:
    - Images from the ARCH Book Set (4,270 pathology images from textbooks)
    - VQA questions from the PathMMU HuggingFace dataset

    Source:
    - PathMMU: https://huggingface.co/datasets/jamessyx/PathMMU
    - ARCH: https://warwick.ac.uk/fac/cross_fac/tia/data/arch/
    """

    _expected_lengths: Dict[str, int] = {
        "val": 80,
        "test": 799,
        "test_tiny": 208,
    }
    """Expected dataset lengths for each split."""

    _arch_book_set_url: str = "https://warwick.ac.uk/fac/cross_fac/tia/data/arch/books_set.zip"
    """URL for downloading the ARCH Book Set images."""

    _arch_book_set_md5: str = "9d0d295ac888d46379ef5fabccaa689d"
    """MD5 hash for validating the ARCH Book Set zip file."""

    _expected_arch_images: int = 4270
    """Expected number of images in the ARCH Book Set."""

    _license: str = "CC-BY-ND-4.0 (PathMMU), CC-BY-NC-SA-4.0 (ARCH Book Set)"
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
        """Initialize the PathMMU Atlas dataset.

        Args:
            root: Path to the root directory for storing dataset files.
            split: Dataset split to use. One of "val", "test", or "test_tiny".
            download: Whether to download the dataset if not found locally. When True,
                downloads both ARCH Book Set images and PathMMU VQA data. When False,
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
    @override
    def classes(self) -> List[str]:
        return ["A", "B", "C", "D", "E"]

    @property
    @override
    def class_to_idx(self) -> Dict[str, int]:
        return {label: idx for idx, label in enumerate(self.classes)}

    @property
    def _path_mmu_hf_path(self) -> str:
        """Returns the path to the PathMMU HuggingFace content directory."""
        return os.path.join(self._root, "path_mmu_hf")

    @property
    def _arch_path(self) -> str:
        """Returns the path to the ARCH dataset directory."""
        return os.path.join(self._root, "arch")

    @override
    def __len__(self) -> int:
        return len(self._samples)

    @override
    def prepare_data(self) -> None:
        """Prepares the dataset by downloading ARCH images and PathMMU VQA data if needed."""
        if self._download:
            self._download_arch_book_set()
            self._download_path_mmu_hf()

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
        # Only strip if ALL options have uppercase letter prefixes (A-E followed by ) or .)
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
        return re.sub(r"^[A-E][)\.]\s*", "", option)

    @override
    def load_images(self, index: int) -> list[tv_tensors.Image]:
        sample = self._samples[index]
        image_path = os.path.join(self._arch_path, sample["source_img"])
        return [io.read_image_as_tensor(image_path)]

    @override
    def load_target(self, index: int) -> int:
        """Returns the correct answer as an integer index (0=A, 1=B, etc.)."""
        sample = self._samples[index]
        # The answer field contains the full answer starting with the letter
        # e.g., "B) Skeletal muscle" -> "B" -> 1
        answer_letter = sample["answer"][0]  # Extract just the letter
        if not (answer_letter.isupper() and answer_letter.isalpha()):
            raise ValueError(
                f"Invalid answer format at index {index}: expected uppercase letter (A-E), "
                f"got '{answer_letter}' from answer '{sample['answer']}'"
            )
        return self.class_to_idx[answer_letter]

    @override
    def load_metadata(self, index: int) -> Dict[str, Any]:
        sample = self._samples[index]
        return {
            "answer": sample["answer"],
            "explanation": sample.get("explanation", ""),
            "source_img": sample["source_img"],
        }

    def _download_arch_book_set(self) -> None:
        """Downloads the ARCH Book Set images if not already present."""
        if self._is_arch_book_set_complete():
            return

        self._print_license()
        os.makedirs(self._arch_path, exist_ok=True)

        zip_path = os.path.join(self._arch_path, "books_set.zip")

        # Check if we already have a valid zip file
        if not utils.check_integrity(zip_path, self._arch_book_set_md5):
            # Remove any corrupted/incomplete download
            if os.path.exists(zip_path):
                logger.info(f"Removing corrupted/incomplete download: {zip_path}")
                os.remove(zip_path)

            logger.info("Downloading ARCH Book Set images...")
            utils.download_url(
                self._arch_book_set_url,
                root=self._arch_path,
                filename="books_set.zip",
                md5=self._arch_book_set_md5,
            )

        logger.info("Extracting ARCH Book Set images...")
        utils.extract_archive(zip_path, self._arch_path, remove_finished=True)

    def _is_arch_book_set_complete(self) -> bool:
        """Checks if ARCH Book Set images are already downloaded and complete."""
        images_path = os.path.join(self._arch_path, "books_set", "images")
        if not os.path.isdir(images_path):
            return False

        image_files = [f for f in os.listdir(images_path) if f.endswith(".png")]
        if len(image_files) != self._expected_arch_images:
            return False

        return True

    def _download_path_mmu_hf(self) -> None:
        """Downloads PathMMU data.json from HuggingFace to path_mmu_hf directory."""
        data_json_path = os.path.join(self._path_mmu_hf_path, "data.json")
        if os.path.exists(data_json_path):
            return

        logger.info("Downloading PathMMU data.json from HuggingFace...")
        huggingface_hub.snapshot_download(
            repo_id="jamessyx/PathMMU",
            repo_type="dataset",
            local_dir=self._path_mmu_hf_path,
            allow_patterns=["data.json"],
        )
        logger.info(f"PathMMU HuggingFace data.json saved to {self._path_mmu_hf_path}")

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

        if "Atlas" not in data:
            raise KeyError(
                f"'Atlas' subset not found in data.json. Available keys: {list(data.keys())}"
            )

        atlas_data = data["Atlas"]

        if self._split not in atlas_data:
            raise KeyError(
                f"Split '{self._split}' not found in Atlas subset. "
                f"Available splits: {list(atlas_data.keys())}"
            )

        return atlas_data[self._split]

    def _print_license(self) -> None:
        """Prints the dataset license."""
        logger.info(f"Dataset license: {self._license}")
