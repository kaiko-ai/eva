"""LLM wrapper for vLLM models."""

from typing import Any, Dict

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
    ) -> None:
        """Initializes the vLLM model wrapper.

        Args:
            model_name_or_path: The model identifier (e.g., a Hugging Face
             repo ID or local path).
            model_kwargs: Additional keyword arguments for initializing the
             vLLM model.
        """
        super().__init__()
        self._model_name_or_path = model_name_or_path
        self._model_kwargs = model_kwargs or {}
        self.load_model()

    @override
    def load_model(self) -> None:
        """Loads the vLLM model and sets up the tokenizer."""
        self._model = LLM(model=self._model_name_or_path, **self._model_kwargs)
        self._tokenizer = self._model.get_tokenizer()

    def _apply_chat_template(self, prompt: str):
        """Converts a prompt string into a TokensPrompt using chat template.

        Args:
            prompt: The input prompt as a string.

        Returns:
            A TokensPrompt object ready for generation.

        Raises:
            ValueError: If the tokenizer does not support a chat template.
        """
        messages = [{"role": "user", "content": prompt}]
        if self._tokenizer.chat_template is None:
            raise ValueError("Tokenizer does not have a chat template.")
        encoded_messages = self._tokenizer.apply_chat_template(
            [messages],
            tokenize=True,
            add_generation_prompt=True,
        )
        if len(encoded_messages[0]) >= 2 and (
            encoded_messages[0][0] == self._tokenizer.bos_token_id
            and encoded_messages[0][1] == self._tokenizer.bos_token_id
        ):
            encoded_messages[0] = encoded_messages[0][1:]
        return [TokensPrompt(prompt_token_ids=encoded_messages[0])]

    def generate(self, prompt: str) -> str:
        """Generates text for the given prompt using the vLLM model.

        Args:
            prompt: A string prompt for generation.

        Returns:
            The generated text response.
        """
        tokens_prompt = self._apply_chat_template(prompt)
        outputs = self._model.generate(tokens_prompt, SamplingParams(**self._model_kwargs))
        return outputs[0].outputs[0].text
