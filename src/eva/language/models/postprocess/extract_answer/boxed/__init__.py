"""API for extracting answers from boxed model outputs."""

from eva.language.models.postprocess.extract_answer.boxed.discrete import (
    ExtractDiscreteAnswerFromBoxed,
)
from eva.language.models.postprocess.extract_answer.boxed.free_form import ExtractAnswerFromBoxed

__all__ = [
    "ExtractAnswerFromBoxed",
    "ExtractDiscreteAnswerFromBoxed",
]
