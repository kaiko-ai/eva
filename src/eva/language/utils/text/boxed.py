"""Boxed text utilities."""

import re
from typing import Dict


def extract_boxed(response: str, raise_if_missing: bool = False) -> Dict[str, str] | None:
    r"""Extracts content from \\boxed{} tags and converts to a dictionary.

    Args:
        response: The input string potentially containing \\boxed{} tags.
        raise_if_missing: Whether to raise an error if no boxed content is found.
            If set to False, will return None instead.

    Returns:
        Dict[str, str] | None: The extracted boxed content as a dictionary with key "answer"
            or None if no boxed content is found and `raise_if_missing` is False.

    Note:
        Nested braces (e.g., \\boxed{\\frac{1}{2}}) are not supported in this implementation.
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

        # Match \boxed{content} - use non-greedy to get first complete match
        pattern = r"\\boxed\{(.*?)\}"
        matches = re.findall(pattern, clean_response, flags=re.DOTALL)

        if len(matches) == 0:
            raise ValueError("No \\boxed{} content found.")
        elif len(matches) > 1:
            raise ValueError(
                f"Multiple \\boxed{{}} expressions found ({len(matches)}). "
                "Cannot determine which answer is correct."
            )
        else:
            answer = matches[0].strip()
            boxed_dict = {"answer": answer}

    except Exception as e:
        if raise_if_missing:
            raise ValueError("Failed to extract boxed content from the response.") from e
        boxed_dict = None

    return boxed_dict
