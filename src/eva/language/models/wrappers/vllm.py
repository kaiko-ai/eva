"""LLM wrapper for vLLM models."""

from typing import Any, Dict, List

from loguru import logger
from typing_extensions import override

try:
    from vllm import LLM, SamplingParams  # type: ignore
    from vllm.inputs import TokensPrompt  # type: ignore
    from vllm.transformers_utils.tokenizer import AnyTokenizer  # type: ignore
except ImportError as e:
    raise ImportError(
        "vLLM is required for VllmModel but not installed. "
        "vLLM must be installed manually as it requires CUDA and is not included in dependencies. "
        "Install with: pip install vllm "
        "Note: vLLM requires Linux with CUDA support for optimal performance. "
        "For alternatives, consider using HuggingFaceModel or LiteLLMModel."
    ) from e

from eva.language.data.messages import MessageSeries
from eva.language.models.typings import TextBatch
from eva.language.models.wrappers import base
from eva.language.utils.text import messages as message_utils


class VllmModel(base.LanguageModel):
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
        system_prompt: str | None = None,
        generation_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        """Initializes the vLLM model wrapper.

        Args:
            model_name_or_path: The model identifier (e.g., a Hugging Face
             repo ID or local path).
            model_kwargs: Arguments required to initialize the vLLM model,
                see [link](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py)
                for more information.
            system_prompt: System prompt to use.
            generation_kwargs: Arguments required to generate the output,
                need to align with the arguments of
                [vllm.SamplingParams](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py).

        """
        super().__init__(system_prompt=system_prompt)
        self._model_name_or_path = model_name_or_path
        self._model_kwargs = model_kwargs or {}
        self._generation_kwargs = generation_kwargs or {}

        # Postpone heavy LLM initialisation to avoid pickling issues
        self._llm_model: LLM | None = None
        self._llm_tokenizer: AnyTokenizer | None = None

    @override
    def load_model(self) -> None:
        """Create the vLLM engine on first use.

        This lazy initialisation keeps the wrapper picklable by Ray / Lightning.
        """
        if self._llm_model is not None:
            return
        self._llm_model = LLM(model=self._model_name_or_path, **self._model_kwargs)
        if self._llm_model is None:
            raise RuntimeError("Model not initialized")
        self._llm_tokenizer = self._llm_model.get_tokenizer()

    def _tokenize_messages(self, messages: List[MessageSeries]) -> List[TokensPrompt]:
        """Apply chat template to the messages.

        Args:
            messages: List of raw user strings.

        Returns:
            List of encoded messages.

        Raises:
            ValueError: If the tokenizer does not have a chat template.
        """
        self.load_model()
        if self._llm_tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")

        if not hasattr(self._llm_tokenizer, "chat_template"):
            raise ValueError("Tokenizer does not have a chat template.")

        chat_messages = list(map(message_utils.format_chat_message, messages))

        encoded_messages = self._llm_tokenizer.apply_chat_template(
            chat_messages,  # type: ignore
            tokenize=True,
            add_generation_prompt=True,
        )

        # Check for double start token (BOS)
        if (
            hasattr(self._llm_tokenizer, "bos_token_id")
            and self._llm_tokenizer.bos_token_id is not None
            and isinstance(encoded_messages, list)
            and len(encoded_messages) >= 2
            and encoded_messages[0] == self._llm_tokenizer.bos_token_id
            and encoded_messages[1] == self._llm_tokenizer.bos_token_id
        ):

            logger.warning("Found a double start token in the input_ids. Removing it.")
            encoded_messages = encoded_messages[1:]

        result = []
        for encoded_message in encoded_messages:
            if isinstance(encoded_message, (list, tuple)):
                # Ensure all elements are integers
                token_ids = [
                    int(token) if isinstance(token, (int, str)) and str(token).isdigit() else 0
                    for token in encoded_message
                ]
            else:
                # Handle single token case
                token_id = (
                    int(encoded_message)
                    if isinstance(encoded_message, (int, str)) and str(encoded_message).isdigit()
                    else 0
                )
                token_ids = [token_id]

            result.append(TokensPrompt(prompt_token_ids=token_ids))

        return result

    @override
    def format_inputs(self, batch: TextBatch) -> List[TokensPrompt]:
        """Formats inputs for vLLM models.

        Args:
            batch: A batch of text and image inputs.

        Returns:
            List of formatted prompts.
        """
        message_batch, _, _ = TextBatch(*batch)
        message_batch = message_utils.batch_insert_system_message(
            message_batch, self.system_message
        )
        message_batch = list(map(message_utils.combine_system_messages, message_batch))

        return self._tokenize_messages(message_batch)

    @override
    def model_forward(self, batch: List[TokensPrompt]) -> List[str]:
        """Generates text for the given prompt using the vLLM model.

        Args:
            batch: A list encoded / tokenized messages (TokensPrompt objects).

        Returns:
            The generated text response.
        """
        self.load_model()
        if self._llm_model is None:
            raise RuntimeError("Model not initialized")

        outputs = self._llm_model.generate(batch, SamplingParams(**self._generation_kwargs))
        return [output.outputs[0].text for output in outputs]
