"""HuggingFace Vision-Language Model Wrapper."""

import functools
from typing import Any, Dict, Literal

import torch
from typing_extensions import override

from eva.language.models import wrappers as language_wrappers
from eva.language.models.typings import ModelOutput, TextBatch
from eva.language.utils.text import messages as language_message_utils
from eva.multimodal.models.typings import TextImageBatch
from eva.multimodal.models.wrappers import base
from eva.multimodal.utils.batch import unpack_batch
from eva.multimodal.utils.text import messages as message_utils


class HuggingFaceModel(base.VisionLanguageModel):
    """Wrapper class for HuggingFace vision-language models."""

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
        """
        super().__init__(system_prompt=system_prompt)

        self.image_key = image_key
        self.image_position: Literal["before_text", "after_text"] = image_position

        self._language_model = language_wrappers.HuggingFaceModel(
            model_name_or_path=model_name_or_path,
            model_class=model_class,
            model_kwargs=model_kwargs,
            system_prompt=system_prompt,
            processor_kwargs=processor_kwargs,
            generation_kwargs=generation_kwargs,
        )

        self.processor = self._language_model.processor
        self.model = self._language_model.model

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
        with_images = image_batch is not None

        message_batch = language_message_utils.batch_insert_system_message(
            message_batch, self.system_message
        )
        message_batch = list(map(language_message_utils.combine_system_messages, message_batch))

        if self.processor.chat_template is not None:  # type: ignore
            templated_text = [
                self.processor.apply_chat_template(  # type: ignore
                    message,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for message in map(
                    functools.partial(
                        message_utils.format_huggingface_message,
                        with_images=with_images,
                        image_position=self.image_position,
                    ),
                    message_batch,
                )
            ]
        else:
            raise NotImplementedError("Currently only chat models are supported.")

        processor_inputs: Dict[str, Any] = {
            "text": templated_text,
            "return_tensors": "pt",
        }

        if with_images:
            processor_inputs[self.image_key] = [[image] for image in image_batch]

        return self.processor(**processor_inputs).to(self.model.device)  # type: ignore

    @override
    def model_forward(self, batch: Dict[str, torch.Tensor]) -> ModelOutput:
        """Generates text output from the model.

        Args:
            batch: A dictionary containing the input data.

        Returns:
            The model output containing generated text.
        """
        return self._language_model.model_forward(batch)
