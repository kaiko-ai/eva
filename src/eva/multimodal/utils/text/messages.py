"""Message formatting utilities for multimodal models."""

from typing import Any, Dict, List

from torchvision import tv_tensors

from eva.language import utils as language_utils
from eva.language.data.messages import MessageSeries
from eva.multimodal.utils import image as image_utils


def format_huggingface_message(
    message: MessageSeries, with_images: bool = False, image_token: str | None = None
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
                            "text": str(item.content).replace(
                                "<image>", image_token or ""
                            ),  # TODO: test this
                        },
                        {"type": "image"},
                    ],
                }
            )
    return formatted_message


def format_litellm_message(
    message: MessageSeries, image: tv_tensors.Image | None
) -> List[Dict[str, Any]]:
    """Format a message series for LiteLLM API.

    Args:
        message: The message series to format.
        image: Optional image to include in the message.

    Returns:
        A list of formatted message dictionaries.
    """
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
                            "text": str(item.content).replace(
                                "<image>", ""
                            ),  # TODO: is this necessary?
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    f"data:image/png;base64,"
                                    f"{image_utils.encode_image(image, encoding='base64')}"
                                )
                            },
                        },
                    ],
                }
            )
    return formatted_message
