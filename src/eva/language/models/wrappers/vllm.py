"""LLM wrapper for vLLM models."""

from typing import Any, Dict, List

from loguru import logger
from typing_extensions import override
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from eva.core.models.wrappers import base


class VLLMTextModel(base.BaseModel):
    """Wrapper class for using vLLM for text generation.

    This wrapper loads a vLLM model, sets up the tokenizer and sampling
    parameters, and uses a chat template to convert a plain string prompt
    into the proper input format for vLLM generation. It then returns the
    generated text response.
    """

    def __init__(
        self,
        model_name_or_path: str,
        model_kwargs: Dict[str, Any] | None = None,
        generation_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        """Initializes the vLLM model wrapper.

        Args:
            model_name_or_path: The model identifier (e.g., a Hugging Face
             repo ID or local path).
            model_kwargs: Arguments required to initialize the vLLM model,
                see https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py
                for more information.
            generation_kwargs: Arguments required to generate the output,
                need to align with the arguments of
                [vllm.SamplingParams](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py).

        """
        super().__init__()
        self._model_name_or_path = model_name_or_path
        self._model_kwargs = model_kwargs or {}
        self._generation_kwargs = generation_kwargs or {}
        self._model = LLM(model=self._model_name_or_path, **self._model_kwargs)
        self._tokenizer = self._model.get_tokenizer()

    @override
    def load_model(self) -> None:
        """Note: The vLLM model needs to be initialized in __init__
        to avoid pickling issues in Ray. """
        pass

    def _apply_chat_template(
        self, messages: list[list[dict[str, str]]]
    ) -> Any:  # Should be vllm.inputs.TokensPrompt
        """Apply chat template to the messages.

        Args:
            messages: List of messages.

        Returns:
            List of encoded messages.

        Raises:
            ValueError: If the tokenizer does not have a chat template.
        """
        if self.tokenizer.chat_template is None:
            raise ValueError("Tokenizer does not have a chat template.")
        encoded_messages = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )

        # Check for double start token
        wrong_sequence = [self.tokenizer.bos_token_id] * 2
        if encoded_messages[: len(wrong_sequence)] == wrong_sequence:
            logger.warning("Found a double start token in the input_ids. Removing it.")
            encoded_messages.pop(0)

        return [
            TokensPrompt(prompt_token_ids=encoded_message) for encoded_message in encoded_messages
        ]

    def generate(self, prompts: List[str]) -> List[str]:
        """Generates text for the given prompt using the vLLM model.

        Args:
            prompt: A string prompt for generation.

        Returns:
            The generated text response.
        """
        prompt_tokens = self._apply_chat_template(prompts)
        outputs = self._model.generate(prompt_tokens, SamplingParams(**self._generation_kwargs))
        return [output[0].outputs[0].text for output in outputs]
