"""LLM wrapper for HuggingFace `transformers` models."""

from typing import Any, Dict, List, Literal

from transformers.pipelines import pipeline
from typing_extensions import override

from eva.core.models.wrappers import base


class HuggingFaceTextModel(base.BaseModel[List[str], List[str]]):
    """Wrapper class for loading HuggingFace `transformers` models using pipelines."""

    def __init__(
        self,
        model_name_or_path: str,
        task: Literal["text-generation"] = "text-generation",
        model_kwargs: Dict[str, Any] | None = None,
        generation_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        """Initializes the model.

        Args:
            model_name_or_path: The model name or path to load the model from.
                This can be a local path or a model name from the `HuggingFace`
                model hub.
            task: The pipeline task. Defaults to "text-generation".
            model_kwargs: Additional arguments for configuring the pipeline.
            generation_kwargs: Additional generation parameters (temperature, max_length, etc.).
        """
        super().__init__()

        self._model_name_or_path = model_name_or_path
        self._task = task
        self._model_kwargs = model_kwargs or {}
        self._generation_kwargs = generation_kwargs or {}

        self.load_model()

    @override
    def load_model(self) -> None:
        """Loads the model as a Hugging Face pipeline."""
        self._pipeline = pipeline(
            task=self._task,
            model=self._model_name_or_path,
            trust_remote_code=True,
            **self._model_kwargs,
        )

    @override
    def model_forward(self, prompts: List[str]) -> List[str]:
        """Generates text using the pipeline.

        Args:
            prompts: The input prompts for the model.

        Returns:
            The generated text as a string.
        """
        outputs = self._pipeline(prompts, return_full_text=False, **self._generation_kwargs)
        if outputs is None:
            raise ValueError("Outputs from the model are None.")
        results = []
        for output in outputs:
            if isinstance(output, list):
                results.append(output[0]["generated_text"])  # type: ignore
            else:
                results.append(output["generated_text"])  # type: ignore
        return results
