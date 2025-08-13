"""HuggingFace Vision-Language Model wrapper."""

import functools
from typing import Any, Dict

import torch
import transformers
from loguru import logger
from typing_extensions import override

from eva.language.utils.text import messages as language_message_utils
from eva.multimodal.models.typings import TextImageBatch
from eva.multimodal.models.wrappers import base
from eva.multimodal.utils.text import messages as message_utils


class HuggingFaceModel(base.VisionLanguageModel):
    """Lightweight wrapper for Huggingface VLMs.

    Args:
        model_name_or_path: The name of the model to use.
        model_class: The class of the model to use.
        model_kwargs: Additional model arguments.
        processor_kwargs: Additional processor arguments.
        generation_kwargs: Additional generation arguments.
    """

    def __init__(
        self,
        model_name_or_path: str,
        model_class: str,
        model_kwargs: Dict[str, Any] | None = None,
        system_prompt: str | None = None,
        processor_kwargs: Dict[str, Any] | None = None,
        generation_kwargs: Dict[str, Any] | None = None,
        image_token: str | None = None,
    ):
        """Initialize the HuggingFace model wrapper.

        Args:
            model_name_or_path: The name or path of the model to use.
            model_class: The class of the model to use.
            model_kwargs: Additional model arguments.
            system_prompt: System prompt to use.
            processor_kwargs: Additional processor arguments.
            generation_kwargs: Additional generation arguments.
            image_token: Token to use for images.
        """
        super().__init__(system_prompt=system_prompt)

        self.model_name_or_path = model_name_or_path
        self.model_kwargs = model_kwargs or {}
        self.base_model_class = model_class
        self.processor_kwargs = processor_kwargs or {}
        self.generation_kwargs = generation_kwargs or {}
        self.image_token = image_token

        self.load_processor()
        self.load_model()

    @override
    def load_model(self) -> None:
        """Setting up the model. Used for delayed model initialization.

        Raises:
            ValueError: If the model class is not found in transformers or if the model
                does not support gradient checkpointing but it is enabled.
        """
        logger.info(f"Configuring model: {self.model_name_or_path}")
        if hasattr(transformers, self.base_model_class):
            model_class = getattr(transformers, self.base_model_class)
        else:
            raise ValueError(f"Model class {self.base_model_class} not found in transformers")

        self.model = model_class.from_pretrained(self.model_name_or_path, **self.model_kwargs)

    def load_processor(self) -> None:
        """Initialize the processor."""
        try:
            self.processor = transformers.AutoProcessor.from_pretrained(
                self.model_name_or_path,
                **self.processor_kwargs,
            )
        except (
            OSError,
            ValueError,
        ) as e:
            raise RuntimeError(f"Could not load processor: {e}") from e

    @override
    def format_inputs(self, batch: TextImageBatch) -> Dict[str, torch.Tensor]:
        """Create the inputs in format expected by the model.

        Args:
            batch: A dictionary containing the input data, which may include:
                - "text": List of messages formatted for the model.
                - "image": List of (PIL) images to be processed.

        Returns:
            A dictionary containing the processed text and image tokens for model input.
        """
        message_batch, image_batch, _, _ = TextImageBatch(*batch)
        with_images = image_batch is not None

        message_batch = language_message_utils.batch_insert_system_message(
            message_batch, self.system_message
        )
        message_batch = list(map(language_message_utils.combine_system_messages, message_batch))

        templated_text = [
            self.processor.apply_chat_template(
                message,
                add_generation_prompt=True,
                tokenize=False,
            )
            for message in map(
                functools.partial(
                    message_utils.format_huggingface_message,
                    with_images=with_images,
                    image_token=self.image_token,
                ),
                message_batch,
            )
        ]

        processor_inputs = {
            "text": templated_text,
            "return_tensors": "pt",
            **self.processor_kwargs,
        }

        if with_images:
            processor_inputs["image"] = [[image] for image in image_batch]

        return self.processor(**processor_inputs).to(self.model.device)

    @override
    def model_forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Generates text output from the model. Is called by the `generate` method.

        Args:
            batch: A dictionary containing the input data, which may include:
                - "text": List of messages formatted for the model.
                - "image": List of image tensors.

        Returns:
            A dictionary containing the processed input and the model's output.
        """
        output = self.model.generate(**batch, **self.generation_kwargs)
        decoded_input, decoded_output = self._decode_output(
            output,
            batch["input_ids"].shape[-1],
        )
        return {
            "processed_input": decoded_input,
            "output": decoded_output,
        }

    def _decode_output(
        self, output: torch.Tensor, instruction_length: int
    ) -> tuple[list[str], list[str]]:
        """Decode the model's output to text.

        Args:
            output: The raw output from the model.
            instruction_length: The length of the instruction part.

        Returns:
            A list of decoded text responses.
        """
        decoded_input = self.processor.batch_decode(
            output[:, :instruction_length], skip_special_tokens=True
        )
        decoded_output = self.processor.batch_decode(
            output[:, instruction_length:], skip_special_tokens=True
        )

        return decoded_input, decoded_output
