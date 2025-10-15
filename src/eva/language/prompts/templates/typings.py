"""Typings for prompt templates."""

from typing_extensions import TypedDict


class QuestionAnswerExample(TypedDict):
    """A question-answer example for few-shot prompting."""

    question: str
    answer: str
