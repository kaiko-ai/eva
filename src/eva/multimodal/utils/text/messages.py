import functools
from typing import Any, Dict, List

from torchvision import tv_tensors

from eva.language import utils as language_utils
from eva.language.data.messages import MessageSeries, SystemMessage
from eva.multimodal.utils import image as image_utils


def format_huggingface_message(
    message: MessageSeries, with_images: bool = False, image_token: str = ""
) -> List[Dict[str, Any]]:
    """Formats a message series into a format suitable for Huggingface models."""
    if not with_images:
        return language_utils.format_message(message)

    formatted_message = []
    for item in message:
        if item.role == "system":
            formatted_message += language_utils.format_message([item])
        else:
            formatted_message.append(
                {
                    "role": item.role,
                    "content": [
                        {
                            "type": "text",
                            "text": str(item.content).replace("<image>", image_token),
                        },
                        {"type": "image"},
                    ],
                }
            )
    return formatted_message


def format_litellm_message(
    message: MessageSeries, image: tv_tensors.Image | None
) -> List[Dict[str, Any]]:
    if image is None:
        return language_utils.format_message(message)

    formatted_message = []
    for item in message:
        if item.role == "system":
            formatted_message += language_utils.format_message([item])
        else:
            formatted_message.append(
                {
                    "role": item.role,
                    "content": [
                        {
                            "type": "text",
                            "text": str(item.content).replace("<image>", ""),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_utils.encode_image(image, encoding='base64')}"
                            },
                        },
                    ],
                }
            )
    return formatted_message


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
    system_messages = list(filter(lambda item: item.role == "system", message))
    if len(system_messages) == 0:
        return message

    combined_content = join_char.join(item.content for item in system_messages)
    non_system_messages = list(filter(lambda item: item.role != "system", message))
    return [SystemMessage(content=combined_content)] + non_system_messages


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


def messages_to_string(messages: List[Dict[str, Any]]) -> str:
    """Extract string content from a list of message dicts"""
    return " ".join(
        f"{message['role'].upper()}: "
        + " ".join(
            chunk["text"] if chunk["type"] == "text" else "" for chunk in message["content"]
        ).strip()
        for message in messages
    ).strip()
