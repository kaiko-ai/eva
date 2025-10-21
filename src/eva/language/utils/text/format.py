"""Text formatting utilities."""

import re
import string
from typing import Literal, Sequence


def format_list_items(
    items: Sequence[str],
    style: Literal["bullets", "letters", "numbers"] = "bullets",
) -> str:
    """Format list items.

    Args:
        items: List of answer items (non-empty strings).
        style: Format type — "bullets", "letters", or "numbers".

    Returns:
        A formatted string with one item per line, prefixed accordingly.
    """
    if not items or not all(isinstance(it, str) and it.strip() for it in items):
        raise ValueError(f"`items` must be all non-empty strings, got {items}.")

    match style:
        case "letters":
            letters = string.ascii_uppercase
            if len(items) > len(letters):
                raise ValueError(f"Maximum {len(letters)} items supported for letter format.")
            return "\n".join(f"{letters[i]}. {it.strip()}" for i, it in enumerate(items))

        case "numbers":
            return "\n".join(f"{i+1}. {it.strip()}" for i, it in enumerate(items))

        case "bullets":
            return "\n".join(f"- {it.strip()}" for it in items)

        case _:
            raise ValueError("Invalid style. Choose from 'bullets', 'letters', or 'numbers'.")


def remove_multi_blank_lines(text: str) -> str:
    """Removes multiple consecutive blank lines from the text.

    Args:
        text: The input text.

    Returns:
        The text with multiple blank lines reduced to single blank lines.
    """
    return re.sub(r"\n\s*\n+", "\n\n", text.strip()) + "\n"
