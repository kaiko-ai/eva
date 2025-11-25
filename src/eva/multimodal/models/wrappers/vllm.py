"""LLM wrapper for vLLM models."""

from typing import Any, Dict, List, Literal

from typing_extensions import override

from eva.multimodal.utils.batch import unpack_batch

try:
    from vllm import LLM, SamplingParams  # type: ignore
    from vllm.inputs import TokensPrompt  # type: ignore
except ImportError as e:
    raise ImportError(
        "vLLM is required for VllmModel but not installed. "
        "vLLM must be installed manually as it requires CUDA and is not included in dependencies. "
        "Install with: pip install vllm "
        "Note: vLLM requires Linux with CUDA support for optimal performance. "
        "For alternatives, consider using HuggingFaceModel or LiteLLMModel."
    ) from e


import torchvision.transforms.functional as F
from transformers import AutoTokenizer

from eva.language.models.constants import MAX_NEW_TOKENS
from eva.language.models.typings import ModelOutput
from eva.language.utils.text import messages as language_message_utils
from eva.multimodal.models.typings import TextImageBatch
from eva.multimodal.models.wrappers import base
from eva.multimodal.utils.text import messages as message_utils


class VllmModel(base.VisionLanguageModel):
    """Wrapper class for using vLLM for text generation.

    This wrapper loads a vLLM model, sets up the tokenizer and sampling
    parameters, and uses a chat template to convert a plain string prompt
    into the proper input format for vLLM generation. It then returns the
    generated text response.
    """

    _default_model_kwargs: Dict[str, Any] = {
        "max_model_len": "32768",
        "gpu_memory_utilization": 95,
        "tensor_parallel_size": 1,
        "dtype": "auto",
    }

    _default_generation_kwargs = {
        "temperature": 0.0,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "top_p": 1.0,
        "top_k": -1,
        "n": 1,
    }

    def __init__(
        self,
        model_name_or_path: str,
        model_kwargs: Dict[str, Any] | None = None,
        system_prompt: str | None = None,
        generation_kwargs: Dict[str, Any] | None = None,
        image_position: Literal["before_text", "after_text"] = "after_text",
    ) -> None:
        """Initializes the vLLM model wrapper.

        Args:
            model_name_or_path: The model identifier (e.g., a HuggingFace repo ID or local path).
            model_kwargs: Arguments required to initialize the vLLM model,
                see [link](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py)
                for more information.
            system_prompt: System prompt to use.
            generation_kwargs: Arguments required to generate the output.
                See [vllm.SamplingParams](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py).
            image_position: Position of the image in the input sequence.

        """
        super().__init__(system_prompt=system_prompt)
        self.model_name_or_path = model_name_or_path
        self.image_position = image_position
        self.model_kwargs = self._default_model_kwargs | (model_kwargs or {})
        self.generation_kwargs = self._default_generation_kwargs | (generation_kwargs or {})

        self.model = LLM(model=self.model_name_or_path, **self.model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

    @override
    def format_inputs(self, batch: TextImageBatch) -> List[Dict[str, Any]]:
        """Formats inputs for vLLM models.

        Args:
            batch: A batch of text and image inputs.

        Returns:
            A list of input dictionaries with "prompt" and "multi_modal_data" keys
            to pass to vLLM's generate method.
        """
        message_batch, image_batch, _, _ = unpack_batch(batch)
        with_images = image_batch is not None

        message_batch = language_message_utils.batch_insert_system_message(
            message_batch, self.system_message
        )
        message_batch = list(map(language_message_utils.combine_system_messages, message_batch))

        input_dicts = []
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None:
            for messages, image in zip(
                message_batch, image_batch or [] * len(message_batch), strict=False
            ):
                pil_image = F.to_pil_image(image)
                formatted_messages = message_utils.format_huggingface_message(
                    messages,
                    with_images=with_images,
                    image_position=self.image_position,
                )
                templated_messages = self.tokenizer.apply_chat_template(
                    formatted_messages,  # type: ignore
                    tokenize=False,
                    add_generation_prompt=True,
                )
                input_dicts.append(
                    {
                        "prompt": templated_messages,
                        "multi_modal_data": {"image": pil_image},
                    }
                )
        else:
            raise NotImplementedError("Currently only chat models are supported.")

        return input_dicts

    @override
    def model_forward(self, batch: List[TokensPrompt]) -> ModelOutput:
        """Generates text for the given prompt using the vLLM model.

        Args:
            batch: A list encoded / tokenized messages (TokensPrompt objects).

        Returns:
            The generated text response.
        """
        outputs = self.model.generate(
            batch, sampling_params=SamplingParams(**self.generation_kwargs)
        )
        output_texts = [output.outputs[0].text for output in outputs]

        return ModelOutput(generated_text=output_texts)
