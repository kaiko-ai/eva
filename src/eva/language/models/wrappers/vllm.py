"""Language model wrapper for vLLM."""

from typing import Any, Dict, List

from typing_extensions import override

try:
    from vllm import LLM, SamplingParams  # type: ignore
except ImportError as e:
    raise ImportError(
        "vLLM is required for VllmModel but not installed. "
        "vLLM must be installed manually as it requires CUDA and is not included in dependencies. "
        "Install with: pip install vllm "
        "Note: vLLM requires Linux with CUDA support for optimal performance. "
        "For alternatives, consider using HuggingFaceModel or LiteLLMModel."
    ) from e

from transformers import AutoTokenizer

from eva.language.models.constants import MAX_NEW_TOKENS
from eva.language.models.typings import ModelOutput, TextBatch
from eva.language.models.wrappers import base
from eva.language.utils.text import messages as message_utils


class VllmModel(base.LanguageModel):
    """Wrapper class for using vLLM for text generation.

    This wrapper loads a vLLM model, sets up the tokenizer and sampling
    parameters, and uses a chat template to format inputs for generation.
    """

    _default_model_kwargs = {
        "max_model_len": 32768,
        "gpu_memory_utilization": 0.95,
        "tensor_parallel_size": 1,
        "dtype": "auto",
        "trust_remote_code": True,
    }

    _default_generation_kwargs = {
        "temperature": 0.0,
        "max_tokens": MAX_NEW_TOKENS,
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
        """
        super().__init__(system_prompt=system_prompt)
        self.model_name_or_path = model_name_or_path
        self.model_kwargs = self._default_model_kwargs | (model_kwargs or {})
        self.generation_kwargs = self._default_generation_kwargs | (generation_kwargs or {})

        self.model: LLM | None = None
        self.tokenizer: Any | None = None

    def configure_model(self):
        """Use configure_model hook to load model in lazy fashion."""
        if self.model is None:
            self.model = self.load_model()
        if self.tokenizer is None:
            self.tokenizer = self.load_tokenizer()

    @override
    def load_model(self) -> LLM:
        """Loads the vLLM model."""
        return LLM(model=self.model_name_or_path, **self.model_kwargs)

    def load_tokenizer(self) -> AutoTokenizer:
        """Loads the tokenizer.

        Raises:
            NotImplementedError: If the tokenizer does not have a chat template.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
            raise NotImplementedError("Currently only chat models are supported.")
        return tokenizer

    @override
    def format_inputs(self, batch: TextBatch) -> List[Dict[str, Any]]:
        """Formats inputs for vLLM models.

        Args:
            batch: A batch of text inputs.

        Returns:
            A list of input dictionaries with "prompt" key.
        """
        message_batch, _, _ = TextBatch(*batch)
        message_batch = message_utils.batch_insert_system_message(
            message_batch, self.system_message
        )
        message_batch = list(map(message_utils.combine_system_messages, message_batch))

        input_dicts = []
        for messages in message_batch:
            formatted_messages = message_utils.format_chat_message(messages)
            templated_messages = self.tokenizer.apply_chat_template(
                formatted_messages,  # type: ignore
                tokenize=False,
                add_generation_prompt=True,
            )
            input_dicts.append({"prompt": templated_messages})

        return input_dicts

    @override
    def model_forward(self, batch: List[Dict[str, Any]]) -> ModelOutput:
        """Generates text using the vLLM model.

        Args:
            batch: A list of input dictionaries containing "prompt" key
                (output of `format_inputs`).

        Returns:
            ModelOutput containing the generated text responses.
        """
        outputs = self.model.generate(
            batch, sampling_params=SamplingParams(**self.generation_kwargs)
        )
        output_texts = [output.outputs[0].text for output in outputs]

        return ModelOutput(generated_text=output_texts)
