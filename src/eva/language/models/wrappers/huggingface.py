"""LLM wrapper for HuggingFace `transformers` models."""

from typing import Any, Dict, Literal

import transformers
from typing_extensions import override

from eva.core.models.wrappers import base


class HuggingFaceTextModel(base.BaseModel):
    """Wrapper class for loading HuggingFace `transformers` models using pipelines."""

    def __init__(
        self,
        model_name_or_path: str,
        task: Literal["text-generation", "text-classification"] = "text-generation",
        model_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        """Initializes the model.

        Args:
            model_name_or_path: The model name or path to load the model from.
                This can be a local path or a model name from the `HuggingFace`
                model hub.
            task: The pipeline task. Defaults to "text-generation".
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
            trust_remote_code=True, **self._model_kwargs
        )

    def generate(self, prompts: list[str], **generate_kwargs) -> Any:
        """Generates text using the pipeline.

        Args:
            prompts: The input prompts for the model.
            generate_kwargs: Additional generation parameters (temperature).

        Returns:
            The generated text as a string.
        """
        outputs = self._pipeline(prompts, return_full_text=False, **generate_kwargs)
        return [
            output[0]["generated_text"] if isinstance(output, list) else output
            for output in outputs
        ]
