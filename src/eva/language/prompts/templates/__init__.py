"""Prompt templating API."""

from eva.language.prompts.templates.base import PromptTemplate
from eva.language.prompts.templates.boxed import (
    BoxedFreeFormQuestionPromptTemplate,
    BoxedMultipleChoicePromptTemplate,
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
from eva.language.prompts.templates.xml import XmlMultipleChoicePromptTemplate

__all__ = [
    "BoxedFreeFormQuestionPromptTemplate",
    "BoxedMultipleChoicePromptTemplate",
    "FreeFormQuestionPromptTemplate",
    "JsonFreeFormQuestionPromptTemplate",
    "JsonMultipleChoicePromptTemplate",
    "MultipleChoicePromptTemplate",
    "PromptTemplate",
    "RawFreeFormQuestionPromptTemplate",
    "RawMultipleChoicePromptTemplate",
    "XmlMultipleChoicePromptTemplate",
]
