"""Utility functions for handling MessageSeries objects."""

from eva.language.data.messages import MessageSeries


def messages_to_string(messages: MessageSeries) -> str:
    """Convert a MessageSeries object to a string.

    Args:
        messages: The MessageSeries object to convert.

    Returns:
        The string representation of the MessageSeries.
    """
    return " ".join(f"{message.role.upper()}: {message.content}" for message in messages)
