"""Factory module for creating answer extractors based on answer format."""

from typing import Literal

from eva.language.models.postprocess.extract_answer.base import ExtractAnswerFromStructuredOutput
from eva.language.models.postprocess.extract_answer.json import (
    ExtractAnswerFromJson,
    ExtractDiscreteAnswerFromJson,
)
from eva.language.models.postprocess.extract_answer.raw import (
    ExtractAnswerFromRaw,
    ExtractDiscreteAnswerFromRaw,
)
from eva.language.models.postprocess.extract_answer.xml import (
    ExtractAnswerFromXml,
    ExtractDiscreteAnswerFromXml,
)


class ExtractDiscreteAnswer:
    """Factory for creating discrete answer extractors."""

    def __new__(
        cls, answer_format: Literal["json", "xml", "raw"], extract_kwargs: dict
    ) -> ExtractAnswerFromStructuredOutput:
        """Create an extractor based on the answer format."""
        match answer_format:
            case "json":
                return ExtractDiscreteAnswerFromJson(**extract_kwargs)
            case "xml":
                return ExtractDiscreteAnswerFromXml(**extract_kwargs)
            case "raw":
                return ExtractDiscreteAnswerFromRaw(**extract_kwargs)
            case _:
                raise ValueError(f"Unknown answer format: {answer_format}")


class ExtractAnswer:
    """Factory for creating answer extractors."""

    def __new__(
        cls, answer_format: Literal["json", "xml", "raw"], extract_kwargs: dict
    ) -> ExtractAnswerFromStructuredOutput:
        """Create an extractor based on the answer format."""
        match answer_format:
            case "json":
                return ExtractAnswerFromJson(**extract_kwargs)
            case "xml":
                return ExtractAnswerFromXml(**extract_kwargs)
            case "raw":
                return ExtractAnswerFromRaw(**extract_kwargs)
            case _:
                raise ValueError(f"Unknown answer format: {answer_format}")
