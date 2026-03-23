"""JSON text utilities."""

import json
import re
from typing import Dict

from json_repair import repair_json


def extract_json(response: str, repair: bool = True, raise_if_missing: bool = False) -> Dict | None:
    """Extracts a JSON object from a string, repairing it if necessary.

    Args:
        response: The input string potentially containing JSON.
        repair: Whether to repair the JSON if it's malformed.
        raise_if_missing: Whether to raise an error if no JSON is found.
            If set to False, will return None instead.

    Returns:
        Dict | None: The extracted JSON object or None if
            no JSON is found and `raise_if_missing` is False.
    """
    try:
        # Strip markdown code blocks and language identifier
        clean_response = re.sub(
            r"```(?:json)?\n(.*?)\n```", r"\1", response, flags=re.DOTALL
        ).strip()
        if repair:
            clean_response = repair_json(clean_response)
        json_response = json.loads(clean_response)
        if not isinstance(json_response, dict):
            raise ValueError(f"Expected a dictionary, got {type(json_response)}.")
    except Exception as e:
        if raise_if_missing:
            raise ValueError("Failed to extract a JSON object from the response.") from e
        json_response = None

    return json_response
