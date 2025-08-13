"""Message formatting utilities for language models."""

from typing import Any, Dict, List

from eva.language.data.messages import MessageSeries


def format_message(message: MessageSeries) -> List[Dict[str, Any]]:
    """Formats a message series into a format following OpenAI's API specification."""
    return [
        {
            "role": item.role,
            "content": [
                {
                    "type": "text",
                    "text": item.content,
                }
            ],
        }
        for item in message
    ]
