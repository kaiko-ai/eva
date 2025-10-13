"""Text formatting utilities."""

import re
import string
from typing import Literal, Sequence


def format_as_bullet_points(
    options: Sequence[str],
    style: Literal["bullets", "letters", "numbers"] = "bullets",
) -> str:
    """Format answer options for inclusion in a prompt.

    Args:
        options: List of answer options (non-empty strings).
        style: Format type — "bullets", "letters", or "numbers".

    Returns:
        A formatted string with one option per line, prefixed accordingly.
    """
    if not options or not all(isinstance(opt, str) and opt.strip() for opt in options):
        raise ValueError(f"`options` must be all non-empty strings, got {options}.")

    match style:
        case "letters":
            letters = string.ascii_uppercase
            if len(options) > len(letters):
                raise ValueError(f"Maximum {len(letters)} options supported for letter format.")
            return "\n".join(f"{letters[i]}. {opt.strip()}" for i, opt in enumerate(options))

        case "numbers":
            return "\n".join(f"{i+1}. {opt.strip()}" for i, opt in enumerate(options))

        case "bullets":
            return "\n".join(f"- {opt.strip()}" for opt in options)

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
