"""Delimiter-based text extraction utilities."""

from typing import List

from loguru import logger

from eva.language.utils.text.raw import _extract_answer_from_options


def extract_delimiter(
    text: str,
    delimiter: str = "####",
    answer_key: str = "answer",
    case_sensitive: bool = False,
    answer_options: List[str] | None = None,
) -> dict | None:
    """Extract answer from text after a delimiter marker.

    Extracts the content that appears after the specified delimiter (default "####").
    For multiple-choice questions, validates the extracted answer against provided options.
    Returns the last occurrence if multiple delimiters are found.

    Args:
        text: The input string containing the delimiter and answer.
        delimiter: The delimiter string to search for (default "####").
        answer_key: The key to use in the returned dictionary.
        case_sensitive: Whether to treat matching as case sensitive.
        answer_options: Optional list of valid answer options for validation.

    Returns:
        Dictionary with answer_key mapped to extracted answer, or None if no delimiter found.

    Examples:
        >>> extract_delimiter("Let me think... #### B")
        {'answer': 'B'}

        >>> extract_delimiter("Analysis: #### The answer is 42", answer_key="result")
        {'result': 'The answer is 42'}

        >>> extract_delimiter("First #### A, but actually #### B")
        {'answer': 'B'}

        >>> extract_delimiter("No delimiter here")
        None
    """
    if not text or not isinstance(text, str):
        return None

    try:
        parts = text.split(delimiter)

        if len(parts) < 2:
            logger.debug(f"No delimiter '{delimiter}' found in text")
            return None

        answer_text = parts[-1].strip()

        if not answer_text:
            logger.warning(f"Empty answer after delimiter '{delimiter}'")
            return None

        if answer_options:
            extracted_answer = _extract_answer_from_options(
                answer_text, answer_options, case_sensitive
            )
            if extracted_answer:
                result = extracted_answer if case_sensitive else extracted_answer.upper()
                return {answer_key: result}

            logger.warning(
                f"Could not match delimited answer '{answer_text}' "
                f"to any of the provided options: {answer_options}"
            )
            return None

        return {answer_key: answer_text}

    except Exception as e:
        logger.warning(f"Failed to extract answer from delimiter text: {e}")
        return None
