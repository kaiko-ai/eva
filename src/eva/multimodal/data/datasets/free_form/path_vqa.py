"""PathVQA free-form question answering dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal

import numpy as np
from datasets import Dataset, load_dataset, load_from_disk
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional
from typing_extensions import override

from eva.language.data.messages import MessageSeries, UserMessage
from eva.language.prompts import templates
from eva.multimodal.data.datasets.schemas import TransformsSchema
from eva.multimodal.data.datasets.text_image import TextImageDataset

_FREE_FORM_LABELS = {"free-form", "freeform", "open", "open-ended", "descriptive"}
_NON_FREE_FORM_LABELS = {
    "yes/no",
    "yesno",
    "binary",
    "boolean",
    "counting",
    "multiple-choice",
    "multiple choice",
    "number",
}


class PathVQAFreeForm(TextImageDataset[str]):
    """Subset of PathVQA dataset with free-form questions."""

    _default_prompt_template = templates.FreeFormQuestionPromptTemplate()
    """Default prompt template for formatting questions and context."""

    _default_render_kwargs = {
        "preamble": (
            "Read the provided question and provide a concise, factual answer based only on what is visible in the image."
        ),
        "context": None,
        "examples": None,  # TODO: add few shot examples?
    }
    """Default kwargs for the template.render() call."""

    def __init__(
        self,
        root: str | None = None,
        split: Literal["train", "val", "test"] | None = None,
        *,
        transforms: TransformsSchema | None = None,
        download: bool = True,
        max_samples: int | None = None,
        prompt_template: templates.PromptTemplate | None = None,
        prompt_render_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        """Initializes the dataset."""
        super().__init__(transforms=transforms)

        if split not in {"train", "val", "test", None}:
            raise ValueError(
                f"Invalid split '{split}'. Expected one of ['train', 'val', 'test', None]."
            )

        self._root = Path(root).expanduser() if root else None
        self._split = split
        self._download = download
        self._max_samples = max_samples

        self.prompt_template = prompt_template or self._default_prompt_template
        self.prompt_render_kwargs = dict(self._default_render_kwargs)
        if prompt_render_kwargs:
            self.prompt_render_kwargs.update(prompt_render_kwargs)

        self.dataset: Dataset | None = None
        self._dataset_cache_path: Path | None = None

    # --------------------------------------------------------------------- #
    # Dataset lifecycle
    # --------------------------------------------------------------------- #

    @override
    def prepare_data(self) -> None:
        """Ensures the dataset is downloaded and cached on disk if configured."""
        dataset_path: Path | None = None

        if self._root:
            dataset_dir = self._root / (self._split or "all")
            dataset_path = dataset_dir / "dataset"
            self._dataset_cache_path = dataset_path

            if dataset_path.exists():
                return

            if not self._download:
                raise FileNotFoundError(
                    f"Cached dataset not found at '{dataset_path}'. "
                    "Provide a valid download directory or enable `download=True`."
                )

            dataset_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._dataset_cache_path = None
            if not self._download:
                # Assume dataset is already present in the HuggingFace cache.
                return

        split_name = self._resolve_hf_split()
        dataset = self._download_dataset(dataset_path, split_name)
        del dataset

    @override
    def configure(self) -> None:
        """Loads the dataset split into memory."""
        dataset_path = self._dataset_cache_path or self._dataset_storage_path()
        split_name = self._resolve_hf_split()
        dataset = self._load_dataset(dataset_path, split_name)
        dataset = self._filter_free_form(dataset)

        if self._max_samples is not None and len(dataset) > self._max_samples:
            dataset = dataset.select(range(self._max_samples))

        self.dataset = dataset

    @override
    def validate(self) -> None:
        if self.dataset is None:
            raise RuntimeError("Dataset has not been configured. Call `setup` first.")
        if len(self.dataset) == 0:
            raise ValueError("PathVQAFreeForm dataset is empty.")

    @override
    def __len__(self) -> int:
        if not self.dataset:
            raise RuntimeError("Dataset has not been prepared. Call `prepare_data` first.")
        return len(self.dataset)

    # --------------------------------------------------------------------- #
    # Map-style dataset API
    # --------------------------------------------------------------------- #

    @override
    def load_text(self, index: int) -> MessageSeries:
        sample = self._get_sample(index)
        question = self._resolve_question(sample)
        if not question:
            raise ValueError(f"No question found for sample at index {index}.")

        render_kwargs = dict(self.prompt_render_kwargs)
        context = self._resolve_context(sample)
        if context is not None:
            render_kwargs["context"] = context

        prompt = self.prompt_template.render(question=question, **render_kwargs)
        return [UserMessage(content=prompt)]

    @override
    def load_image(self, index: int) -> tv_tensors.Image:
        sample = self._get_sample(index)
        image = sample.get("image")
        if image is None:
            image_path = sample.get("image_path") or sample.get("image_path_absolute")
            if not image_path:
                raise FileNotFoundError(f"No image reference found for sample at index {index}.")
            image = Image.open(image_path)
        if isinstance(image, Image.Image):
            image_array = np.array(image.convert("RGB"))
        elif isinstance(image, dict):
            if "array" in image:
                image_array = np.asarray(image["array"], dtype=np.uint8)
            elif "path" in image:
                with Image.open(image["path"]) as pil_image:
                    image_array = np.array(pil_image.convert("RGB"))
            else:
                raise ValueError(f"Unsupported image dictionary format for sample {index}.")
        else:
            image_array = np.asarray(image, dtype=np.uint8)
        return functional.to_image(image_array)

    @override
    def load_target(self, index: int) -> str:
        sample = self._get_sample(index)
        answer = sample.get("answer") or sample.get("Answer")
        if isinstance(answer, str) and answer:
            return answer
        answers = sample.get("answers") or sample.get("Answers") or []
        if isinstance(answers, list) and answers:
            first = answers[0]
            if isinstance(first, str):
                return first
            if isinstance(first, dict):
                value = first.get("answer") or first.get("text")
                if isinstance(value, str):
                    return value
        raise ValueError(f"No answer found for sample at index {index}.")

    @override
    def load_metadata(self, index: int) -> Dict[str, Any]:
        sample = self._get_sample(index)
        metadata: Dict[str, Any] = {}
        field_mapping = {
            "answer_type": "answer_type",
            "answerType": "answer_type",
            "question_type": "question_type",
            "questionType": "question_type",
            "image_id": "image_id",
            "imageId": "image_id",
            "question_id": "question_id",
            "questionId": "question_id",
        }
        for source_key, target_key in field_mapping.items():
            value = sample.get(source_key)
            if value is not None:
                metadata[target_key] = value

        if "image_path" in sample:
            metadata["image_path"] = sample["image_path"]
        if "context" in sample:
            metadata.setdefault("context", sample["context"])
        metadata["split"] = self._split
        return metadata

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    def _dataset_storage_path(self) -> Path | None:
        if not self._root:
            return None
        return self._root / (self._split or "all") / "dataset"

    def _download_dataset(self, dataset_path: Path | None, split: str) -> Dataset:
        dataset = load_dataset(
            "flaviagiammarino/path-vqa",
            split=split,
            download_mode="reuse_dataset_if_exists",
        )

        if dataset_path is not None:
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(dataset_path))

        return dataset  # type: ignore[return-value]

    def _load_dataset(self, dataset_path: Path | None, split: str) -> Dataset:
        if dataset_path and dataset_path.exists():
            return load_from_disk(str(dataset_path))

        return load_dataset(
            "flaviagiammarino/path-vqa",
            split=split,
            download_mode="reuse_dataset_if_exists",
        )  # type: ignore[return-value]

    def _filter_free_form(self, dataset: Dataset) -> Dataset:
        def is_free_form(example: Dict[str, Any]) -> bool:
            answer_type = example.get("answer_type") or example.get("answerType")
            if not isinstance(answer_type, str):
                return True
            normalized = answer_type.strip().lower().replace("_", "-")
            if normalized in _NON_FREE_FORM_LABELS:
                return False
            if normalized in _FREE_FORM_LABELS:
                return True
            return not any(keyword in normalized for keyword in ("yes", "no", "multiple", "count"))

        return dataset.filter(is_free_form)

    def _resolve_hf_split(self) -> str:
        if self._split is None:
            return "train+validation+test"
        if self._split == "val":
            return "validation"
        return self._split

    def _get_sample(self, index: int) -> Dict[str, Any]:
        if not self.dataset:
            raise RuntimeError("Dataset has not been configured. Call `setup` first.")
        if index < 0 or index >= len(self.dataset):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.dataset)}.")
        return dict(self.dataset[index])

    def _resolve_question(self, sample: Dict[str, Any]) -> str | None:
        for key in ("question", "Question", "prompt"):
            value = sample.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _resolve_context(self, sample: Dict[str, Any]) -> Any | None:
        for key in ("context", "Context", "facts", "Facts"):
            value = sample.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, list) and value:
                return value
        return None
