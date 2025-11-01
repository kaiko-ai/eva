"""API for extracting answers from model outputs."""

from eva.language.models.postprocess.extract_answer.factory import (
    ExtractAnswer,
    ExtractDiscreteAnswer,
)

__all__ = ["ExtractAnswer", "ExtractDiscreteAnswer"]
