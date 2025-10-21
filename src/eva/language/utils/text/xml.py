"""XML text utilities."""

import re
import xml.etree.ElementTree as ET
from typing import Dict


def extract_xml(response: str, raise_if_missing: bool = False) -> Dict[str, str] | None:
    """Extracts XML tags from a string and converts them to a dictionary.

    Args:
        response: The input string potentially containing XML tags.
        raise_if_missing: Whether to raise an error if no XML is found.
            If set to False, will return None instead.

    Returns:
        Dict[str, str] | None: The extracted XML tags as a flat dictionary or None if
            no XML is found and `raise_if_missing` is False.
    """
    try:
        # First, try to extract from markdown code blocks
        code_fence_match = re.search(r"```(?:xml)?\s*\n(.*?)\n```", response, flags=re.DOTALL)
        if code_fence_match:
            clean_response = code_fence_match.group(1).strip()
        else:
            # If no code fence, try to find XML tags in the text
            # Extract all complete XML elements
            xml_tags = re.findall(r"<(\w+)>(.*?)</\1>", response, flags=re.DOTALL)
            if xml_tags:
                # Reconstruct XML string with all tags
                clean_response = "".join(f"<{tag}>{content}</{tag}>" for tag, content in xml_tags)
            else:
                clean_response = response.strip()

        # Wrap in a temporary root element to handle multiple top-level tags
        wrapped_xml = f"<root>{clean_response}</root>"

        # Parse the XML
        root = ET.fromstring(wrapped_xml)

        # Convert XML to a flat dictionary (only direct children)
        xml_dict = {}
        for child in root:
            # Use the tag name as the key and text content as the value
            xml_dict[child.tag] = child.text.strip() if child.text else ""

        if not xml_dict:
            raise ValueError("No XML tags found.")

    except Exception as e:
        if raise_if_missing:
            raise ValueError("Failed to extract an XML object from the response.") from e
        xml_dict = None

    return xml_dict
