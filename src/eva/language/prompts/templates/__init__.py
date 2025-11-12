"""Prompt templating API."""

from eva.language.prompts.templates.base import PromptTemplate
from eva.language.prompts.templates.delimiter import (
    DelimiterFreeFormQuestionPromptTemplate,
    DelimiterMultipleChoicePromptTemplate,
)
from eva.language.prompts.templates.factory import (
    FreeFormQuestionPromptTemplate,
    MultipleChoicePromptTemplate,
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

__all__ = [
    "PromptTemplate",
    "FreeFormQuestionPromptTemplate",
    "MultipleChoicePromptTemplate",
    "JsonFreeFormQuestionPromptTemplate",
    "JsonMultipleChoicePromptTemplate",
    "RawFreeFormQuestionPromptTemplate",
    "RawMultipleChoicePromptTemplate",
    "XmlFreeFormQuestionPromptTemplate",
    "XmlMultipleChoicePromptTemplate",
    "DelimiterFreeFormQuestionPromptTemplate",
    "DelimiterMultipleChoicePromptTemplate",
]
