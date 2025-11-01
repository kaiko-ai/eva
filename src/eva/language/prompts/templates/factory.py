"""Prompt templates for evaluation."""

from typing import Literal

from eva.language.prompts.templates.base import PromptTemplate
from eva.language.prompts.templates.json import (
    JsonFreeFormQuestionPromptTemplate,
    JsonMultipleChoicePromptTemplate,
)
from eva.language.prompts.templates.raw import (
    RawFreeFormQuestionPromptTemplate,
    RawMultipleChoicePromptTemplate,
)
from eva.language.prompts.templates.xml import (
    XmlFreeFormQuestionPromptTemplate,
    XmlMultipleChoicePromptTemplate,
)


class FreeFormQuestionPromptTemplate(PromptTemplate):
    """Factory for free-form question prompt templates based on answer format."""

    def __new__(
        cls, answer_format: Literal["json", "xml", "raw"], **template_kwargs
    ) -> PromptTemplate:
        """Create a free-form question prompt template based on the answer format."""
        match answer_format:
            case "json":
                return JsonFreeFormQuestionPromptTemplate(**template_kwargs)
            case "xml":
                return XmlFreeFormQuestionPromptTemplate(**template_kwargs)
            case "raw":
                return RawFreeFormQuestionPromptTemplate(**template_kwargs)
            case _:
                raise ValueError(f"Unknown answer format: {answer_format}")


class MultipleChoicePromptTemplate(PromptTemplate):
    """Factory for Multiple Choice QA prompt templates based on answer format."""

    def __new__(
        cls, answer_format: Literal["json", "xml", "raw"], **template_kwargs
    ) -> PromptTemplate:
        """Create a multiple-choice prompt template based on the answer format."""
        match answer_format:
            case "json":
                return JsonMultipleChoicePromptTemplate(**template_kwargs)
            case "xml":
                return XmlMultipleChoicePromptTemplate(**template_kwargs)
            case "raw":
                return RawMultipleChoicePromptTemplate(**template_kwargs)
            case _:
                raise ValueError(f"Unknown answer format: {answer_format}")
