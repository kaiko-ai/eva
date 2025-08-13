"""PatchCamelyon dataset with text prompts for multimodal classification."""

from typing import Any, Dict, Literal

from torchvision import tv_tensors
from typing_extensions import override

from eva.language.data.messages import MessageSeries, UserMessage
from eva.multimodal.data.datasets.schemas import TransformsSchema
from eva.multimodal.data.datasets.text_image import TextImageDataset
from eva.vision.data import datasets as vision_datasets


class PatchCamelyon(TextImageDataset[str], vision_datasets.PatchCamelyon):
    """PatchCamelyon image classification using a multiple choice text prompt."""

    _default_prompt = (
        "You are a pathology expert helping pathologists to analyze images of tissue samples. "
        "Does this image show metastatic breast tissue? Answer with a single letter A: no, B: yes "
        "Please always provide an answer, even if you are not sure. Answer: "
    )

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"],
        download: bool = False,
        transforms: TransformsSchema | None = None,
        prompt: str | None = None,
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
        """
        super().__init__(root=root, split=split, download=download, transforms=transforms)

        self.prompt = prompt or self._default_prompt
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

    @property
    @override
    def class_to_idx(self) -> Dict[str, int]:
        return {"A": 0, "B": 1}

    @override
    def load_text(self, index: int) -> MessageSeries:
        return [UserMessage(content=self.prompt)]

    @override
    def load_image(self, index: int) -> tv_tensors.Image:
        return vision_datasets.PatchCamelyon.load_data(self, index)

    @override
    def load_target(self, index: int) -> str:
        target = int(vision_datasets.PatchCamelyon.load_target(self, index).item())
        return self.idx_to_class[target]

    @override
    def load_metadata(self, index: int) -> Dict[str, Any] | None:
        return vision_datasets.PatchCamelyon.load_metadata(self, index)
