"""Prompt templates for evaluation."""

from typing import Literal

from typing_extensions import override

from eva.language.prompts.templates.base import PromptTemplate
from eva.language.prompts.templates.delimiter import (
    DelimiterFreeFormQuestionPromptTemplate,
    DelimiterMultipleChoicePromptTemplate,
)
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
        cls, answer_format: Literal["json", "xml", "raw", "delimiter"], **template_kwargs
    ) -> PromptTemplate:
        """Create a free-form question prompt template based on the answer format.

        Args:
            answer_format: The format to use for answers ('json', 'xml', 'raw', or 'delimiter').
            **template_kwargs: Keyword arguments passed to the template constructor.

        Returns:
            An appropriate prompt template instance for the specified format.
        """
        match answer_format:
            case "json":
                return JsonFreeFormQuestionPromptTemplate(**template_kwargs)
            case "xml":
                return XmlFreeFormQuestionPromptTemplate(**template_kwargs)
            case "raw":
                return RawFreeFormQuestionPromptTemplate(**template_kwargs)
            case "delimiter":
                return DelimiterFreeFormQuestionPromptTemplate(**template_kwargs)
            case _:
                raise ValueError(f"Unknown answer format: {answer_format}")

    @override
    def render(self, **kwargs) -> str:
        raise NotImplementedError("Factory class should not be instantiated directly.")


class MultipleChoicePromptTemplate(PromptTemplate):
    """Factory for Multiple Choice QA prompt templates based on answer format."""

    def __new__(
        cls, answer_format: Literal["json", "xml", "raw", "delimiter"], **template_kwargs
    ) -> PromptTemplate:
        """Create a multiple-choice prompt template based on the answer format.

        Args:
            answer_format: The format to use for answers ('json', 'xml', 'raw', or 'delimiter').
            **template_kwargs: Keyword arguments passed to the template constructor.

        Returns:
            An appropriate prompt template instance for the specified format.
        """
        match answer_format:
            case "json":
                return JsonMultipleChoicePromptTemplate(**template_kwargs)
            case "xml":
                return XmlMultipleChoicePromptTemplate(**template_kwargs)
            case "raw":
                return RawMultipleChoicePromptTemplate(**template_kwargs)
            case "delimiter":
                return DelimiterMultipleChoicePromptTemplate(**template_kwargs)
            case _:
                raise ValueError(f"Unknown answer format: {answer_format}")

    @override
    def render(self, **kwargs) -> str:
        raise NotImplementedError("Factory class should not be instantiated directly.")
