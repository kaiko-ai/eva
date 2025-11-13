"""API for extracting answers from raw model outputs."""

from eva.language.models.postprocess.extract_answer.raw.discrete import ExtractDiscreteAnswerFromRaw
from eva.language.models.postprocess.extract_answer.raw.free_form import ExtractAnswerFromRaw

__all__ = [
    "ExtractDiscreteAnswerFromRaw",
    "ExtractAnswerFromRaw",
]
