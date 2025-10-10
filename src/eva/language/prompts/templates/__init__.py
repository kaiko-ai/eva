"""Prompt templating API."""

from eva.language.prompts.templates.base import PromptTemplate
from eva.language.prompts.templates.json import JsonMultipleChoicePromptTemplate

__all__ = ["PromptTemplate", "JsonMultipleChoicePromptTemplate"]
