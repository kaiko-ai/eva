"""Raw text extraction utilities."""

import re

from loguru import logger


def extract_raw(
    text: str,
    answer_options: list[str],
    answer_key: str = "answer",
    case_sensitive: bool = False,
    tail_length: int = 10,
) -> dict | None:
    """Extract multiple-choice answer from model response text using regex patterns.

    Extracts answers by focusing on the last `tail_length` words of the response
    where models typically place final answers. The function cleans the text
    (removes extra whitespace and asterisks), then applies multiple regex patterns
    to find answer options. Returns the last match found to prioritize final answers.

    Supported answer patterns:
    - Explicit markers: "answer: A", "choice: B"
    - Standalone options: "...reasoning. C", "D.", "E"
    - Prefixed answers: "the correct answer is B", "the right choice is C"
    - Options with punctuation: "(A)", "B:", "C."

    Handles both single-character (A, B, C) and multi-character (Yes, No) options.

    Args:
        text: The input string containing the model answer.
        answer_options: List of valid options to match against.
        answer_key: The key to use in the returned dictionary.
        case_sensitive: Whether to treat matching as case sensitive.
        tail_length: Number of words from the end to examine for extraction.

    Returns:
        Dictionary with answer_key mapped to extracted option, or None if no match.

    Examples:
        >>> extract_raw("After analysis, my answer is B")
        {'answer': 'B'}

        >>> extract_raw("The answer is: a")
        {'answer': 'A'}

        >>> extract_raw("Therefore, the answer is Yes", answer_options=["Yes", "No"])
        {'answer': 'Yes'}

        >>> extract_raw("My choice is B", answer_key="selected_option")
        {'selected_option': 'B'}

        >>> extract_raw("I think A but actually B", tail_length=5)
        {'answer': 'B'}

        >>> extract_raw("The answer is **B**")
        {'answer': 'B'}

        >>> extract_raw("Some text without an answer")
        None
    """
    if not text or not isinstance(text, str):
        return None

    try:
        text = text.strip()
        tail_text = _get_text_tail(text, max_words=tail_length)
        cleaned_text = _clean_string(tail_text)
        extracted_answer = _extract_answer_from_options(
            cleaned_text, answer_options, case_sensitive
        )

        if extracted_answer:
            result = extracted_answer if case_sensitive else extracted_answer.upper()
            return {answer_key: result}

        return None

    except Exception as e:
        logger.warning(f"Failed to extract answer from text: {e}")
        return None


def _get_text_tail(text: str, max_words: int = 10) -> str:
    """Get the last `max_words` words from the input text."""
    words = text.split()
    return " ".join(words[-max_words:]) if words else text


def _clean_string(string: str) -> str:
    """Clean the input string by removing extra whitespace and special characters."""
    if not string:
        return ""

    string = re.sub(r"\s+", " ", string)
    string = re.sub(r"\*", "", string)
    return string.strip()


def _extract_answer_from_options(
    text: str, options: list[str], case_sensitive: bool = False
) -> str | None:
    """Extract the final answer from multiple-choice text.

    Args:
        text: Cleaned text to extract answer from.
        options: List of valid options.
        case_sensitive: Whether to treat matching as case sensitive.

    Returns:
        The extracted option or None if no valid answer found.
    """
    if not text:
        return None

    # Distinguish between single-character and multi-character options
    all_single_char = all(len(opt) == 1 for opt in options)
    esc_opts = [re.escape(opt) for opt in options]
    option_regex = f"([{''.join(esc_opts)}])" if all_single_char else f"({'|'.join(esc_opts)})"
    patterns = [
        rf"(?:answer|choice|select)\s*:?\s*[\"']?{option_regex}[\"']?\b",
        rf"(?:^|\s)[\"']?{option_regex}[\"']?[.:]?\s*$",
        rf"(?:correct|right)\s+(?:answer|choice|option)\s*:?\s*[\"']?{option_regex}[\"']?\b",
        rf"(?:^|\s)[\"']?{option_regex}[\"']?(?=\s*[.:)!]|$)",
    ]

    flags = re.MULTILINE if case_sensitive else re.IGNORECASE | re.MULTILINE
    for pattern in patterns:
        matches = re.finditer(pattern, text, flags)
        match_list = list(matches)

        if match_list:
            return match_list[-1].group(1)

    return None
