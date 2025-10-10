"""Typings for prompt templates."""

from typing_extensions import List, NotRequired, TypedDict


class QuestionAnswerExample(TypedDict):
    """A question-answer example for few-shot prompting."""

    question: str
    answer: str
    context: NotRequired[str | List[str]]
