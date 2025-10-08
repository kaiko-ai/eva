"""JSON text utilities."""

import json
import re
from typing import Dict

from json_repair import repair_json


def extract_json(response: str, repair: bool = True) -> Dict:
    """Util func to parse JSON output from LLM models."""
    # Strip markdown code blocks and language identifier
    clean_response = re.sub(r"```(?:json)?\n(.*?)\n```", r"\1", response, flags=re.DOTALL).strip()

    if repair:
        clean_response = repair_json(clean_response)

    return json.loads(clean_response)
