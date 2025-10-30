"""JSON prompt templates for evaluation."""

from eva.language.prompts.templates.json.free_form import (
    JsonFreeFormPromptTemplate,
)
from eva.language.prompts.templates.json.multiple_choice import (
    JsonMultipleChoicePromptTemplate,
)

__all__ = ["JsonMultipleChoicePromptTemplate", "JsonFreeFormPromptTemplate"]
