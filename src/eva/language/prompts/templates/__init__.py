"""Prompt templating API."""

from eva.language.prompts.templates.base import PromptTemplate

from eva.language.prompts.templates.factory import (
    FreeFormQuestionPromptTemplate,
    MultipleChoicePromptTemplate,
)
from eva.language.prompts.templates.json import (
    JsonFreeFormPromptTemplate,
    JsonMultipleChoicePromptTemplate,
)
from eva.language.prompts.templates.raw import (
    RawFreeFormQuestionPromptTemplate,
    RawMultipleChoicePromptTemplate,
)
from eva.language.prompts.templates.xml import XmlMultipleChoicePromptTemplate

__all__ = [
    "JsonMultipleChoicePromptTemplate",
    "RawMultipleChoicePromptTemplate",
    "XmlMultipleChoicePromptTemplate",
    "JsonFreeFormPromptTemplate",
    "RawFreeFormQuestionPromptTemplate",
    "FreeFormQuestionPromptTemplate",
    "MultipleChoicePromptTemplate",
    "PromptTemplate", 
    "JsonMultipleChoicePromptTemplate",
    "FreeFormQuestionPromptTemplate"]
