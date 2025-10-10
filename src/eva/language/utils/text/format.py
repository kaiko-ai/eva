"""Text formatting utilities."""

import re
from typing import Sequence


def format_as_bullet_points(content: str | Sequence[str]) -> str:
    """Formats the given content as bullet points.

    Args:
        content: A string or a sequence of strings to format as bullet points.

    Returns:
        The formatted bullet point string.
    """
    if not isinstance(content, list):
        content = [content]  # type: ignore
    return "\n".join(f"- {item.strip()}" for item in content if item.strip())


def remove_multi_blank_lines(text: str) -> str:
    """Removes multiple consecutive blank lines from the text.

    Args:
        text: The input text.

    Returns:
        The text with multiple blank lines reduced to single blank lines.
    """
    return re.sub(r"\n\s*\n+", "\n\n", text.strip()) + "\n"
