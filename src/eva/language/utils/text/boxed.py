"""Boxed text utilities."""

import re
from typing import Dict


def _find_matching_brace(text: str, start: int) -> int:
    """Find the position of the closing brace matching the opening brace at start.

    Args:
        text: The input string.
        start: Position to start searching from (should be right after an opening brace).

    Returns:
        int: Position of the matching closing brace, or -1 if not found.
    """
    count = 1
    i = start
    while i < len(text) and count > 0:
        if text[i] == "{":
            count += 1
        elif text[i] == "}":
            count -= 1
        i += 1
    return i - 1 if count == 0 else -1


def extract_boxed(response: str, raise_if_missing: bool = False) -> Dict[str, str] | None:
    r"""Extracts content from \\boxed{} tags and converts to a dictionary.

    Args:
        response: The input string potentially containing \\boxed{} tags.
        raise_if_missing: Whether to raise an error if no boxed content is found.
            If set to False, will return None instead.

    Returns:
        Dict[str, str] | None: The extracted boxed content as a dictionary with key "answer"
            or None if no boxed content is found and `raise_if_missing` is False.
            When multiple \\boxed{} expressions are found, returns the last one.

    Note:
        Supports nested braces (e.g., \\boxed{\\frac{1}{2}}) and uses the last
        \\boxed{} expression when multiple are present.
    """
    try:
        # Check if response is wrapped in code fences (latex/math)
        code_fence_match = re.search(
            r"```(?:latex|math)?\s*\n(.*?)\n```", response, flags=re.DOTALL
        )
        if code_fence_match:
            clean_response = code_fence_match.group(1).strip()
        else:
            clean_response = response.strip()

        # Find last \boxed{
        boxed_start = clean_response.rfind("\\boxed{")
        if boxed_start == -1:
            raise ValueError("No \\boxed{} content found.")

        # Find the content between the braces
        content_start = boxed_start + 7  # len('\\boxed{')
        closing_brace = _find_matching_brace(clean_response, content_start)

        if closing_brace == -1:
            raise ValueError("No matching closing brace found.")

        answer = clean_response[content_start:closing_brace].strip()
        boxed_dict = {"answer": answer}

    except Exception as e:
        if raise_if_missing:
            raise ValueError("Failed to extract boxed content from the response.") from e
        boxed_dict = None

    return boxed_dict
