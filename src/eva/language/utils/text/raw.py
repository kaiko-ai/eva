"""Raw text extraction utilities."""

import re

from loguru import logger


def extract_raw(value: str, options: list[str] | None = None) -> dict | None:
    """Extract multiple-choice answer from model response text.

    Focuses on the last few lines of the response where models typically place their final answer.

    Args:
        value: The input string containing the model's response.
        options: Optional list of valid multiple-choice options (e.g., ["A", "B", "C"]).
                If None, will attempt to extract any single letter options.

    Returns:
        Dict[str, str] | None: Dictionary with 'answer' key containing the extracted option,
                              or None if extraction failed.
    """
    if not value or not isinstance(value, str):
        return None

    try:
        text = value.strip()
        tail_text = _get_text_tail(text)
        cleaned_text = _clean_string(tail_text)
        extracted_answer = _extract_answer_from_options(cleaned_text, options)

        if extracted_answer:
            return {"answer": extracted_answer.upper()}

        return None

    except Exception as e:
        logger.warning(f"Failed to extract answer from text: {e}")
        return None


def _get_text_tail(text: str, max_chars: int = 100) -> str:
    """Get the last `max_chars` characters from the input text."""
    match = re.search(r".{0," + str(max_chars) + r"}$", text, flags=re.DOTALL)
    return match.group(0) if match else text


def _clean_string(string: str) -> str:
    """Clean the input string by removing extra whitespace and special characters."""
    if not string:
        return ""

    string = re.sub(r"\s+", " ", string)
    string = re.sub(r"\*", "", string)
    return string.strip()


def _extract_answer_from_options(text: str, options: list[str] | None = None) -> str | None:
    """Extract the final answer from multiple-choice text.

    Args:
        text: Cleaned text to extract answer from.
        options: List of valid options. If None, defaults to A-Z.

    Returns:
        The extracted option or None if no valid answer found.
    """
    if not text:
        return None

    if options is None:
        options = [chr(i) for i in range(ord("A"), ord("Z") + 1)]

    # Distinguish between single-character and multi-character options
    all_single_char = all(len(opt) == 1 for opt in options)
    option_regex = (
        f"([{''.join(re.escape(opt.lower()) for opt in options)}])"
        if all_single_char
        else f"({'|'.join(re.escape(opt.lower()) for opt in options)})"
    )

    patterns = [
        rf"(?:answer|choice)\s*:?\s*{option_regex}\b",
        rf"(?:^|\s){option_regex}[.:]?\s*$",
        rf"(?:correct|right)\s+(?:answer|choice|option)\s*:?\s*{option_regex}\b",
        rf"(?:^|\s){option_regex}(?=\s*[.:]|$)",
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        match_list = list(matches)

        if match_list:
            return match_list[-1].group(1).upper()

    return None
