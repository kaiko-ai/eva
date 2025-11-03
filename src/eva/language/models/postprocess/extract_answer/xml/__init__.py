"""API for extracting answers from XML model outputs."""

from eva.language.models.postprocess.extract_answer.xml.discrete import ExtractDiscreteAnswerFromXml
from eva.language.models.postprocess.extract_answer.xml.free_form import ExtractAnswerFromXml

__all__ = [
    "ExtractAnswerFromXml",
    "ExtractDiscreteAnswerFromXml",
]
