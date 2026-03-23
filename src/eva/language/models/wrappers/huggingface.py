"""LLM wrapper for HuggingFace `transformers` models."""

from typing import Any, Callable, Dict, List, Tuple

import torch
import transformers
from loguru import logger
from torch import nn
from typing_extensions import override

from eva.language.models.constants import MAX_NEW_TOKENS
from eva.language.models.typings import ModelOutput, TextBatch
from eva.language.models.wrappers import base
from eva.language.utils.text import messages as message_utils


class HuggingFaceModel(base.LanguageModel):
    """Wrapper class for loading HuggingFace `transformers` models."""

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
        chat_template: str | None = None,
    ) -> None:
        """Initializes the model.

        Args:
            model_name_or_path: The model name or path to load the model from.
                This can be a local path or a model name from the `HuggingFace`
                model hub.
            model_class: The class of the model to use (e.g., "AutoModelForCausalLM").
            model_kwargs: Additional arguments for configuring the model.
            system_prompt: System prompt to use.
            processor_kwargs: Additional processor/tokenizer arguments.
            generation_kwargs: Additional generation parameters (temperature, max_length, etc.).
            chat_template: Optional chat template name to use with the processor. If None,
                will use the template stored in the checkpoint's processor config.
        """
        super().__init__(system_prompt=system_prompt)

        self.model_name_or_path = model_name_or_path
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.processor_kwargs = processor_kwargs or {}
        self.generation_kwargs = self._default_generation_kwargs | (generation_kwargs or {})
        self.chat_template = chat_template

        self.model: nn.Module
        self.processor: Callable

    def configure_model(self) -> None:
        """Use configure_model hook to load model in lazy fashion."""
        logger.info(f"Configuring model: {self.model_name_or_path}")
        if not hasattr(self, "model"):
            self.model = self.load_model()
        if not hasattr(self, "processor"):
            self.processor = self.load_processor()

    @override
    def load_model(self) -> nn.Module:
        """Loads the model from HuggingFace.

        Raises:
            ValueError: If the model class is not found in transformers or if the model
                does not support generation.
        """
        import transformers  # Reimport here, in case module was modified at runtime by user

        if hasattr(transformers, self.model_class):
            model_class = getattr(transformers, self.model_class)
        else:
            raise ValueError(f"Model class {self.model_class} not found in transformers")

        model = model_class.from_pretrained(self.model_name_or_path, **self.model_kwargs)

        if not hasattr(model, "generate"):
            raise ValueError(f"Model {self.model_name_or_path} does not support generation.")

        return model

    def load_processor(self) -> Callable:
        """Initialize the processor.

        Note: For text-only models, AutoProcessor returns the tokenizer.
        """
        processor = transformers.AutoProcessor.from_pretrained(
            self.processor_kwargs.pop("model_name_or_path", self.model_name_or_path),
            **self.processor_kwargs,
        )
        if self.chat_template is not None:
            processor.chat_template = self.chat_template  # type: ignore
        # To ensure correct generation with batched inputs of different lengths
        if "CausalLM" in self.model_class or "ConditionalGeneration" in self.model_class:
            processor.padding_side = "left"
        # Some older models don't have a padding token by default
        if hasattr(processor, "pad_token") and processor.pad_token is None:
            processor.pad_token = processor.eos_token
        return processor

    @override
    def format_inputs(self, batch: TextBatch) -> Dict[str, torch.Tensor]:
        """Formats inputs for HuggingFace models.

        Note: If multiple system messages are present, they will be combined
        into a single message, given that many models only support a single
        system prompt.

        Args:
            batch: A batch of text inputs.

        Returns:
            A dictionary produced by the tokenizer following a format like:
            {
                "input_ids": ...,
                "attention_mask": ...,
            }
        """
        message_batch, _, _ = TextBatch(*batch)
        message_batch = message_utils.batch_insert_system_message(
            message_batch, self.system_message
        )
        message_batch = list(map(message_utils.combine_system_messages, message_batch))

        if self.processor.chat_template is not None:  # type: ignore
            templated_text = [
                self.processor.apply_chat_template(  # type: ignore
                    message_utils.format_chat_message(message),
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for message in message_batch
            ]
        else:
            templated_text = list(map(message_utils.merge_message_contents, message_batch))

        processor_inputs = {
            "text": templated_text,
            "return_tensors": "pt",
            "padding": True,
            **self.processor_kwargs,
        }

        return self.processor(**processor_inputs).to(self.model.device)  # type: ignore

    @override
    def model_forward(self, batch: Dict[str, torch.Tensor]) -> ModelOutput:
        """Generates text using the model.

        Args:
            batch: A dictionary containing the tokenized input data.

        Returns:
            The model output containing generated text.
        """
        output_ids = self.model.generate(**batch, **self.generation_kwargs)  # type: ignore
        decoded_input, decoded_output = self._decode_ids(output_ids, batch["input_ids"].shape[-1])

        return ModelOutput(
            generated_text=decoded_output,
            input_text=decoded_input,
            output_ids=output_ids,
            attention_mask=batch.get("attention_mask"),
        )

    def _decode_ids(
        self, output: torch.Tensor, instruction_length: int
    ) -> Tuple[List[str], List[str]]:
        """Decode the model's batch input and output to text.

        Args:
            output: The raw output from the model.
            instruction_length: The length of the instruction in the input.

        Returns:
            A tuple containing two lists, the decoded input and output texts.
        """
        decoded_input = self.processor.batch_decode(  # type: ignore
            output[:, :instruction_length], skip_special_tokens=False
        )
        decoded_output = self.processor.batch_decode(  # type: ignore
            output[:, instruction_length:], skip_special_tokens=True
        )

        logger.debug(f"Decoded input: {decoded_input}")
        logger.debug(f"Decoded output: {decoded_output}")

        return decoded_input, decoded_output
