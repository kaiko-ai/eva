"""Factory module for creating answer extractors based on answer format."""

from typing import Literal

from eva.language.models.postprocess.extract_answer.base import ExtractAnswerFromStructuredOutput
from eva.language.models.postprocess.extract_answer.boxed import (
    ExtractAnswerFromBoxed,
    ExtractDiscreteAnswerFromBoxed,
)
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
        cls, answer_format: Literal["json", "xml", "boxed", "raw"], extract_kwargs: dict
    ) -> ExtractAnswerFromStructuredOutput:
        """Create a discrete answer extractor based on the answer format.

        Args:
            answer_format: The format of the answer to extract ('json', 'xml', 'boxed', or 'raw').
            extract_kwargs: Keyword arguments passed to the extractor constructor.

        Returns:
            An extractor instance for the specified format.
        """
        match answer_format:
            case "json":
                return ExtractDiscreteAnswerFromJson(**extract_kwargs)
            case "xml":
                return ExtractDiscreteAnswerFromXml(**extract_kwargs)
            case "raw":
                return ExtractDiscreteAnswerFromRaw(**extract_kwargs)
            case "boxed":
                return ExtractDiscreteAnswerFromBoxed(**extract_kwargs)
            case _:
                raise ValueError(f"Unknown answer format: {answer_format}")


class ExtractAnswer:
    """Factory for creating answer extractors."""

    def __new__(
        cls, answer_format: Literal["json", "xml", "boxed", "raw"], extract_kwargs: dict
    ) -> ExtractAnswerFromStructuredOutput:
        """Create an answer extractor based on the answer format.

        Args:
            answer_format: The format of the answer to extract ('json', 'xml', 'boxed', or 'raw').
            extract_kwargs: Keyword arguments passed to the extractor constructor.

        Returns:
            An appropriate extractor instance for the specified format.
        """
        match answer_format:
            case "json":
                return ExtractAnswerFromJson(**extract_kwargs)
            case "xml":
                return ExtractAnswerFromXml(**extract_kwargs)
            case "raw":
                return ExtractAnswerFromRaw(**extract_kwargs)
            case "boxed":
                return ExtractAnswerFromBoxed(**extract_kwargs)
            case _:
                raise ValueError(f"Unknown answer format: {answer_format}")
