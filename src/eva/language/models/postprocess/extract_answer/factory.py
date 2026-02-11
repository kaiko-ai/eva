"""Factory module for creating answer extractors based on answer format."""

from typing import Any, Dict, List, Literal, Union

import torch
from typing_extensions import TypedDict

from eva.language.models.postprocess.extract_answer.base import (
    ExtractAnswerFromStructuredOutput,
    ExtractDiscreteAnswerFromStructuredOutput,
)
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


class ExtractAnswerKwargs(TypedDict, total=False):
    """Keyword arguments for answer extractors."""

    answer_key: str
    case_sensitive: bool
    raise_if_missing: bool
    missing_answer: str | None
    missing_limit: int
    return_dict: bool


class ExtractDiscreteAnswerKwargs(TypedDict, total=False):
    """Keyword arguments for discrete answer extractors."""

    mapping: Dict[str, int]
    answer_key: str
    case_sensitive: bool
    raise_if_missing: bool
    missing_answer: int
    missing_limit: int


class ExtractDiscreteAnswer:
    """Factory class for creating discrete answer extractors.

    This wrapper class delegates to the appropriate extractor based on answer_format.
    It uses __init__ + __call__ pattern so jsonargparse can properly validate it
    as a callable class.
    """

    _extractor: ExtractDiscreteAnswerFromStructuredOutput

    def __init__(
        self,
        answer_format: Literal["json", "xml", "boxed", "raw"],
        extract_kwargs: ExtractDiscreteAnswerKwargs | None = None,
    ) -> None:
        """Initialize the extractor.

        Args:
            answer_format: The format of the answer to extract ('json', 'xml', 'boxed', or 'raw').
            extract_kwargs: Keyword arguments passed to the extractor constructor.
        """
        if extract_kwargs is None:
            extract_kwargs = {}

        match answer_format:
            case "json":
                self._extractor = ExtractDiscreteAnswerFromJson(**extract_kwargs)
            case "xml":
                self._extractor = ExtractDiscreteAnswerFromXml(**extract_kwargs)
            case "raw":
                self._extractor = ExtractDiscreteAnswerFromRaw(**extract_kwargs)
            case "boxed":
                self._extractor = ExtractDiscreteAnswerFromBoxed(**extract_kwargs)
            case _:
                raise ValueError(f"Unknown answer format: {answer_format}")

    def __call__(self, values: Union[str, List[str]]) -> torch.Tensor:
        """Extract discrete answers from the input values.

        Args:
            values: Input string(s) containing answer data.

        Returns:
            Tensor of discrete answer indices.
        """
        return self._extractor(values)


class ExtractAnswer:
    """Factory class for creating answer extractors.

    This wrapper class delegates to the appropriate extractor based on answer_format.
    It uses __init__ + __call__ pattern so jsonargparse can properly validate it
    as a callable class.
    """

    _extractor: ExtractAnswerFromStructuredOutput

    def __init__(
        self,
        answer_format: Literal["json", "xml", "boxed", "raw"],
        extract_kwargs: ExtractAnswerKwargs | None = None,
    ) -> None:
        """Initialize the extractor.

        Args:
            answer_format: The format of the answer to extract ('json', 'xml', 'boxed', or 'raw').
            extract_kwargs: Keyword arguments passed to the extractor constructor.
        """
        if extract_kwargs is None:
            extract_kwargs = {}

        match answer_format:
            case "json":
                self._extractor = ExtractAnswerFromJson(**extract_kwargs)
            case "xml":
                self._extractor = ExtractAnswerFromXml(**extract_kwargs)
            case "raw":
                self._extractor = ExtractAnswerFromRaw(**extract_kwargs)
            case "boxed":
                self._extractor = ExtractAnswerFromBoxed(**extract_kwargs)
            case _:
                raise ValueError(f"Unknown answer format: {answer_format}")

    def __call__(self, values: Union[str, List[str]]) -> Any:
        """Extract answers from the input values.

        Args:
            values: Input string(s) containing answer data.

        Returns:
            List of extracted answers or dictionaries.
        """
        return self._extractor(values)
