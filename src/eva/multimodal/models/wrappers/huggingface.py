"""HuggingFace Vision-Language Model Wrapper."""

import functools
from typing import Any, Callable, Dict, List

import torch
import transformers
from loguru import logger
from torch import nn
from typing_extensions import override

from eva.language.models.typings import ModelOutput, TextBatch
from eva.language.utils.text import messages as language_message_utils
from eva.multimodal.models.typings import TextImageBatch
from eva.multimodal.models.wrappers import base
from eva.multimodal.utils.batch import unpack_batch
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

    _default_generation_kwargs = {
        "temperature": 0.0,
        "max_new_tokens": 1024,
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
        """
        super().__init__(system_prompt=system_prompt)

        self.model_name_or_path = model_name_or_path
        self.model_kwargs = model_kwargs or {}
        self.base_model_class = model_class
        self.processor_kwargs = processor_kwargs or {}
        self.generation_kwargs = self._default_generation_kwargs | (generation_kwargs or {})
        self.image_key = image_key

        self.processor = self.load_processor()
        self.model = self.load_model()

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
                    ),
                    message_batch,
                )
            ]
        else:
            raise NotImplementedError("Currently only chat models are supported.")

        processor_inputs = {
            "text": templated_text,
            "return_tensors": "pt",
            **self.processor_kwargs,
        }

        if with_images:
            processor_inputs[self.image_key] = [[image] for image in image_batch]

        return self.processor(**processor_inputs).to(self.model.device)  # type: ignore

    @override
    def model_forward(self, batch: Dict[str, torch.Tensor]) -> ModelOutput:
        """Generates text output from the model. Is called by the `generate` method.

        Args:
            batch: A dictionary containing the input data, which may include:
                - "text": List of messages formatted for the model.
                - "image": List of image tensors.

        Returns:
            A dictionary containing the processed input and the model's output.
        """
        output_ids = self.model.generate(**batch, **self.generation_kwargs)  # type: ignore

        return ModelOutput(
            generated_text=self._decode_output(output_ids, batch["input_ids"].shape[-1]),
            input_ids=batch.get("input_ids"),
            output_ids=output_ids,
            attention_mask=batch.get("attention_mask"),
        )

    @override
    def load_model(self) -> nn.Module:
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

        model = model_class.from_pretrained(self.model_name_or_path, **self.model_kwargs)

        if not hasattr(model, "generate"):
            raise ValueError(f"Model {self.model_name_or_path} does not support generation. ")

        return model

    def load_processor(self) -> Callable:
        """Initialize the processor."""
        return transformers.AutoProcessor.from_pretrained(
            self.processor_kwargs.pop("model_name_or_path", self.model_name_or_path),
            **self.processor_kwargs,
        )

    def _decode_output(self, output: torch.Tensor, instruction_length: int) -> List[str]:
        """Decode the model's batch output to text.

        Args:
            output: The raw output from the model.
            instruction_length: The length of the instruction in the input.

        Returns:
            A list of decoded text responses.
        """
        decoded_input = self.processor.batch_decode(  # type: ignore
            output[:, :instruction_length], skip_special_tokens=True
        )
        decoded_output = self.processor.batch_decode(  # type: ignore
            output[:, instruction_length:], skip_special_tokens=True
        )

        logger.debug(f"Decoded input: {decoded_input}")
        logger.debug(f"Decoded output: {decoded_output}")

        return decoded_output
