"""Postprocessing transforms for extracting answers from XML responses."""

from typing import Dict

from typing_extensions import override

from eva.language.models.postprocess.extract_answer.base import (
    ExtractDiscreteAnswerFromStructuredOutput,
)
from eva.language.utils.text import xml as xml_utils


class ExtractDiscreteAnswerFromXml(ExtractDiscreteAnswerFromStructuredOutput):
    """Extracts discrete answers from XML responses and casts them to int tensors."""

    @override
    def _extract_structured_data(self, value: str) -> Dict[str, str] | None:
        """Extract XML data from a string.

        Args:
            value: The input string containing XML.

        Returns:
            Dict[str, str] | None: The extracted XML object or None if extraction failed.
        """
        return xml_utils.extract_xml(value)
