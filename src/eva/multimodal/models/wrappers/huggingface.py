"""HuggingFace Vision-Language Model Wrapper."""

from typing import Any, Callable, Dict, Literal

import torch
from typing_extensions import override

from eva.language.models import wrappers as language_wrappers
from eva.language.models.constants import MAX_NEW_TOKENS
from eva.language.models.typings import ModelOutput, TextBatch
from eva.language.utils.text import messages as language_message_utils
from eva.multimodal.models.typings import TextImageBatch
from eva.multimodal.models.wrappers import base
from eva.multimodal.utils.batch import unpack_batch
from eva.multimodal.utils.text import messages as message_utils


class HuggingFaceModel(base.VisionLanguageModel):
    """Wrapper class for HuggingFace vision-language models."""

    _default_generation_kwargs = {
        "temperature": 0.0,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "top_p": 1.0,
    }
    """Default HF model parameters for evaluation."""

    def __init__(
        self,
        model_name_or_path: str,
        model_class: str,
        model_kwargs: Dict[str, Any] | None = None,
        system_prompt: str | None = None,
        processor_kwargs: Dict[str, Any] | None = None,
        generation_kwargs: Dict[str, Any] | None = None,
        image_key: str = "image",
        image_position: Literal["before_text", "after_text"] = "after_text",
        chat_template: str | None = None,
    ):
        """Initialize the HuggingFace model wrapper.

        Args:
            model_name_or_path: The name or path of the model to use.
            model_class: The class of the model to use.
            model_kwargs: Additional model arguments.
            system_prompt: System prompt to use.
            processor_kwargs: Additional processor arguments.
            generation_kwargs: Additional generation arguments.
            image_key: The key used for image inputs in the chat template.
            image_position: Position of the image in the input sequence.
            chat_template: Optional chat template name to use with the processor. If None,
                will use the template stored in the checkpoint's processor config.
        """
        super().__init__(system_prompt=system_prompt)

        self.image_key = image_key
        self.image_position: Literal["before_text", "after_text"] = image_position
        self.model_name_or_path = model_name_or_path
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.processor_kwargs = processor_kwargs or {}
        self.generation_kwargs = self._default_generation_kwargs | (generation_kwargs or {})
        self.chat_template = chat_template

        self.model: language_wrappers.HuggingFaceModel
        self.processor: Callable

    def configure_model(self) -> None:
        """Use configure_model hook to load model in lazy fashion."""
        if not hasattr(self, "model"):
            self.model = self.load_model()
            self.model.configure_model()
        if not hasattr(self, "processor"):
            self.processor = self.model.processor

    @override
    def load_model(self) -> language_wrappers.HuggingFaceModel:
        return language_wrappers.HuggingFaceModel(
            model_name_or_path=self.model_name_or_path,
            model_class=self.model_class,
            model_kwargs=self.model_kwargs,
            processor_kwargs=self.processor_kwargs,
            generation_kwargs=self.generation_kwargs,
            chat_template=self.chat_template,
        )

    @override
    def format_inputs(self, batch: TextImageBatch | TextBatch) -> Dict[str, torch.Tensor]:
        """Formats inputs for HuggingFace models.

        Args:
            batch: A batch of text and image inputs.

        Returns:
            A dictionary produced by the provided processor following a format like:
            {
                "input_ids": ...,
                "attention_mask": ...,
                "pixel_values": ...
            }
        """
        message_batch, image_batch, _, _ = unpack_batch(batch)

        message_batch = language_message_utils.batch_insert_system_message(
            message_batch, self.system_message
        )
        message_batch = list(map(language_message_utils.combine_system_messages, message_batch))

        if image_batch is None:
            image_batch = [None] * len(message_batch)

        if self.processor.chat_template is not None:  # type: ignore
            templated_text = [
                self.processor.apply_chat_template(  # type: ignore
                    message_utils.format_huggingface_message(
                        message,
                        images=images,
                        image_position=self.image_position,
                    ),
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for message, images in zip(message_batch, image_batch, strict=True)
            ]
        else:
            raise NotImplementedError("Currently only chat models are supported.")

        processor_inputs: Dict[str, Any] = {
            "text": templated_text,
            "return_tensors": "pt",
            **self.processor_kwargs,
        }

        if any(image_batch):
            processor_inputs[self.image_key] = image_batch

        return self.processor(**processor_inputs).to(self.model.model.device)  # type: ignore

    @override
    def model_forward(self, batch: Dict[str, torch.Tensor]) -> ModelOutput:
        """Generates text output from the model.

        Args:
            batch: A dictionary containing the input data.

        Returns:
            The model output containing generated text.
        """
        return self.model.model_forward(batch)
