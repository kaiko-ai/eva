"""Init file for JSON answer extraction postprocessing models."""

from eva.language.models.postprocess.extract_answer.json.discrete import (
    ExtractDiscreteAnswerFromJson,
)
from eva.language.models.postprocess.extract_answer.json.freeform import (
    ExtractAnswerFromJson,
)

__all__ = [
    "ExtractAnswerFromJson",
    "ExtractDiscreteAnswerFromJson",
]
