"""Vision-language model wrapper for vLLM."""

import os
from typing import Any, Dict, List, Literal

from typing_extensions import override

from eva.language.models.constants import MAX_NEW_TOKENS
from eva.multimodal.utils.batch import unpack_batch

try:
    from vllm import LLM, SamplingParams  # type: ignore
except ImportError as e:
    raise ImportError(
        "vLLM is required for using VllmModel but is not installed."
        "Please install with: `pip install vllm`"
    ) from e


import torch
import torchvision.transforms.functional as F
from loguru import logger
from transformers import AutoTokenizer

from eva.language.models.typings import ModelOutput
from eva.language.utils.text import messages as language_message_utils
from eva.multimodal.models.typings import TextImageBatch
from eva.multimodal.models.wrappers import base
from eva.multimodal.utils.text import messages as message_utils


class VllmModel(base.VisionLanguageModel):
    """Wrapper class for vision-language models using vLLM.

    This wrapper supports both text-only and multimodal (text + image) inputs.
    It loads a vLLM model, sets up the tokenizer and sampling parameters,
    and uses a chat template to format inputs for generation.
    """

    _default_model_kwargs = {
        "max_model_len": 32768,
        "gpu_memory_utilization": 0.9,
        "tensor_parallel_size": 1,
        "trust_remote_code": True,
        "enforce_eager": True,  #  reduce start-up time & potential compilation issues
    }

    _default_generation_kwargs = {
        "temperature": 0.0,
        "max_tokens": MAX_NEW_TOKENS,
        "top_p": 1.0,
        "top_k": -1,
        "n": 1,
        "seed": torch.initial_seed(),
    }

    def __init__(
        self,
        model_name_or_path: str,
        model_kwargs: Dict[str, Any] | None = None,
        system_prompt: str | None = None,
        generation_kwargs: Dict[str, Any] | None = None,
        image_position: Literal["before_text", "after_text"] = "after_text",
        chat_template: str | None = None,
    ) -> None:
        """Initializes the vLLM model wrapper.

        Args:
            model_name_or_path: The model identifier (e.g., a HuggingFace repo ID or local path).
                Note that the model must be compatible with vLLM.
            model_kwargs: Arguments required to initialize the vLLM model,
                see [link](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py)
                for more information.
            system_prompt: System prompt to use.
            generation_kwargs: Arguments required to generate the output.
                See [vllm.SamplingParams](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py).
            image_position: Position of the image in the input sequence.
            chat_template: Optional chat template name to use with the tokenizer. If None,
                will use the template stored in the checkpoint's tokenizer config.
        """
        super().__init__(system_prompt=system_prompt)
        self.model_name_or_path = model_name_or_path
        self.image_position: Literal["before_text", "after_text"] = image_position
        self.model_kwargs = self._default_model_kwargs | (model_kwargs or {})
        self.generation_kwargs = self._default_generation_kwargs | (generation_kwargs or {})
        self.chat_template = chat_template

        self.model: LLM
        self.tokenizer: AutoTokenizer

    def configure_model(self) -> None:
        """Use configure_model hook to load model in lazy fashion."""
        if not hasattr(self, "model"):
            self.model = self.load_model()
        if not hasattr(self, "tokenizer"):
            self.tokenizer = self.load_tokenizer()

    @override
    def load_model(self) -> LLM:
        """Loads the vLLM model."""
        logger.info(f"Loading model with kwargs: {self.model_kwargs}")
        logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        return LLM(model=self.model_name_or_path, **self.model_kwargs)

    def load_tokenizer(self) -> AutoTokenizer:
        """Loads the tokenizer.

        Raises:
            NotImplementedError: If the tokenizer does not have a chat template.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        if self.chat_template is not None:
            tokenizer.chat_template = self.chat_template  # type: ignore
        if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
            raise NotImplementedError("Currently only chat models are supported.")
        return tokenizer

    @override
    def format_inputs(self, batch: TextImageBatch) -> List[Dict[str, Any]]:
        """Formats inputs for vLLM models.

        Args:
            batch: A batch of text and optional image inputs.

        Returns:
            A list of input dictionaries with "prompt" key and optional
            "multi_modal_data" key (when images are present).
        """
        message_batch, image_batch, _, _ = unpack_batch(batch)
        with_images = image_batch is not None

        message_batch = language_message_utils.batch_insert_system_message(
            message_batch, self.system_message
        )
        message_batch = list(map(language_message_utils.combine_system_messages, message_batch))

        input_dicts = []
        for messages, images in zip(
            message_batch, image_batch or [None] * len(message_batch), strict=False
        ):
            formatted_messages = message_utils.format_huggingface_message(
                messages,
                images=images if with_images else None,
                image_position=self.image_position,
            )
            templated_messages = self.tokenizer.apply_chat_template(  # type: ignore
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_dict: Dict[str, Any] = {"prompt": templated_messages}
            if images:
                input_dict["multi_modal_data"] = {"image": [F.to_pil_image(img) for img in images]}
            input_dicts.append(input_dict)

        return input_dicts

    @override
    def model_forward(self, batch: List[Dict[str, Any]]) -> ModelOutput:
        """Generates text using the vLLM model.

        Args:
            batch: A list of input dictionaries containing "prompt" and
                optional "multi_modal_data" keys (output of `format_inputs`).

        Returns:
            ModelOutput containing the generated text responses.
        """
        outputs = self.model.generate(
            batch, sampling_params=SamplingParams(**self.generation_kwargs)
        )
        output_texts = [output.outputs[0].text for output in outputs]

        return ModelOutput(generated_text=output_texts)
