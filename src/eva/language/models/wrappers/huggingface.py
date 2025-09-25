"""LLM wrapper for HuggingFace `transformers` models."""

from typing import Any, Callable, Dict, List, Literal

from transformers.pipelines import pipeline
from typing_extensions import override

from eva.language.models.typings import ModelOutput, TextBatch
from eva.language.models.wrappers import base
from eva.language.utils.text import messages as message_utils


class HuggingFaceModel(base.LanguageModel):
    """Wrapper class for loading HuggingFace `transformers` models using pipelines."""

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
        task: Literal["text-generation"] = "text-generation",
        model_kwargs: Dict[str, Any] | None = None,
        system_prompt: str | None = None,
        generation_kwargs: Dict[str, Any] | None = None,
        chat_mode: bool = True,
    ) -> None:
        """Initializes the model.

        Args:
            model_name_or_path: The model name or path to load the model from.
                This can be a local path or a model name from the `HuggingFace`
                model hub.
            task: The pipeline task. Defaults to "text-generation".
            model_kwargs: Additional arguments for configuring the pipeline.
            system_prompt: System prompt to use.
            generation_kwargs: Additional generation parameters (temperature, max_length, etc.).
            chat_mode: Whether the specified model expects chat style messages. If set to False
                the model is assumed to be a standard text completion model and will expect
                plain text string inputs.
        """
        super().__init__(system_prompt=system_prompt)

        self._model_name_or_path = model_name_or_path
        self._task = task
        self._model_kwargs = model_kwargs or {}
        self._generation_kwargs = self._default_generation_kwargs | (generation_kwargs or {})
        self._chat_mode = chat_mode

        self.model = self.load_model()

    @override
    def load_model(self) -> Callable:
        """Loads the model as a Hugging Face pipeline."""
        return pipeline(
            task=self._task,
            model=self._model_name_or_path,
            trust_remote_code=True,
            **self._model_kwargs,
        )

    @override
    def format_inputs(self, batch: TextBatch) -> List[List[Dict[str, Any]]] | List[str]:
        """Formats inputs for HuggingFace models.

        Note: If multiple system messages are present, they will be combined
        into a single message, given that many models only support a single
        system prompt.

        Args:
            batch: A batch of text and image inputs.

        Returns:
            When in chat mode, returns a batch of message series following
            OpenAI's API format {"role": "user", "content": "..."}, for non-chat
            models returns a list of plain text strings.
        """
        message_batch, _, _ = TextBatch(*batch)
        message_batch = message_utils.batch_insert_system_message(
            message_batch, self.system_message
        )
        message_batch = list(map(message_utils.combine_system_messages, message_batch))

        if self._chat_mode:
            return list(map(message_utils.format_chat_message, message_batch))
        else:
            return list(map(message_utils.merge_message_contents, message_batch))

    @override
    def model_forward(self, prompts: List[str]) -> ModelOutput:
        """Generates text using the pipeline.

        Args:
            prompts: The input prompts for the model.

        Returns:
            The generated text as a string.
        """
        outputs = self.model(prompts, return_full_text=False, **self._generation_kwargs)
        if outputs is None:
            raise ValueError("Outputs from the model are None.")

        results = []
        for output in outputs:
            if isinstance(output, list):
                results.append(output[0]["generated_text"])  # type: ignore
            else:
                results.append(output["generated_text"])  # type: ignore

        return ModelOutput(generated_text=results)
