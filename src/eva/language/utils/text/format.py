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