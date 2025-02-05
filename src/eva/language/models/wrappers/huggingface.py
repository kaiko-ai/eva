"""LLM wrapper for HuggingFace `transformers` models."""

from typing import Any, Callable, Dict, Optional
import transformers
from typing_extensions import override
from eva.core.models.wrappers import base


class HuggingFaceTextModel(base.BaseModel):
    """Wrapper class for loading HuggingFace `transformers` models using pipelines."""

    def __init__(
        self,
        model_name_or_path: str,
        task: str = "text-generation",
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initializes the model.

        Args:
            model_name_or_path: The model name or path to load the model from.
                This can be a local path or a model name from the `HuggingFace`
                model hub.
            task: The pipeline task (e.g., "text-generation", "text-classification").
                Defaults to "text-generation".
            model_kwargs: Additional arguments for configuring the pipeline.
        """
        super().__init__()

        self._model_name_or_path = model_name_or_path
        self._task = task
        self._model_kwargs = model_kwargs or {}

        self.load_model()

    @override
    def load_model(self) -> None:
        """Loads the model as a Hugging Face pipeline."""
        self._pipeline = transformers.pipeline(
            task=self._task,
            model=self._model_name_or_path,
            **self._model_kwargs
        )

    def generate(self, prompt: str, **generate_kwargs) -> str:
        """Generates text using the pipeline.

        Args:
            prompt: The input prompt for the model.
            generate_kwargs: Additional generation parameters (e.g., max_length).

        Returns:
            The generated text as a string.
        """
        output = self._pipeline(prompt, return_full_text=False, **generate_kwargs)
        print('!output:', output)
        print('!prompt:', prompt)
        return output[0]["generated_text"] if isinstance(output, list) else output
