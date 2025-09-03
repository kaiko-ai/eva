"""Message formatting utilities for language models."""

import functools
import json
from typing import Any, Dict, List

from eva.language.data.messages import (
    AssistantMessage,
    MessageSeries,
    Role,
    SystemMessage,
    UserMessage,
)


def format_chat_message(message: MessageSeries) -> List[Dict[str, Any]]:
    """Formats a message series into a format following OpenAI's API specification."""
    return [{"role": item.role, "content": item.content} for item in message]


def combine_system_messages(message: MessageSeries, join_char: str = "\n") -> MessageSeries:
    """Combine system messages into a single message.

    This is useful when the MessageSeries contains multiple system messages such
    as `ModelSystemMessage` and `TaskSystemMessage`. But given that most models / apis
    expect a single system message, this function can be used to combines them into one.

    Args:
        message: The message series containing one or multiple messages.
        join_char: The character to use to join the system messages. Default is newline.

    Returns:
        A new message series with system messages combined into one and the
        remaining messages unchanged.
    """
    system_messages = list(filter(lambda item: item.role == Role.SYSTEM, message))
    if len(system_messages) == 0:
        return message

    non_system_messages = list(filter(lambda item: item.role != Role.SYSTEM, message))
    return [
        SystemMessage(content=merge_message_contents(system_messages, join_char=join_char))
    ] + non_system_messages


def merge_message_contents(message: MessageSeries, join_char: str = "\n") -> str:
    """Merges the all contents within a message series into a string.

    Args:
        message: The message series to combine.
        join_char: The character to use to join the message contents. Default is newline.

    Returns:
        A string containing the combined message contents.
    """
    return join_char.join(item.content for item in message)


def insert_system_message(
    message: MessageSeries, system_message: SystemMessage | None
) -> MessageSeries:
    """Insert a system message at the beginning of the message series."""
    if system_message is None:
        return message
    return [system_message] + message


def batch_insert_system_message(
    messages: List[MessageSeries], system_message: SystemMessage | None
) -> List[MessageSeries]:
    """Insert a system message at the beginning of each message series in a batch."""
    return list(
        map(functools.partial(insert_system_message, system_message=system_message), messages)
    )


def serialize(messages: MessageSeries) -> str:
    """Serialize a MessageSeries object into a JSON string.

    Args:
        messages: A list of message objects (MessagesSeries).

    Returns:
        A JSON string representing the message series, with the following format:
            [{"role": "user", "content": "Hello"}, ...]
    """
    serialized_messages = format_chat_message(messages)
    return json.dumps(serialized_messages)


def deserialize(messages: str) -> MessageSeries:
    """Convert a json string to a MessageSeries object.

    Format: [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]
    """
    message_dicts = json.loads(messages)

    message_series = []
    for message_dict in message_dicts:
        if "role" not in message_dict or "content" not in message_dict:
            raise ValueError("`role` or `content` keys are missing.")

        match message_dict["role"]:
            case Role.USER:
                message_series.append(UserMessage(**message_dict))
            case Role.ASSISTANT:
                message_series.append(AssistantMessage(**message_dict))
            case Role.SYSTEM:
                message_series.append(SystemMessage(**message_dict))
            case _:
                raise ValueError(f"Unknown role: {message_dict['role']}")

    return message_series
