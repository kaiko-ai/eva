"""Message formatting utilities for multimodal models."""

import os
from typing import Any, Dict, List, Literal

from torchvision import tv_tensors

from eva.language import utils as language_utils
from eva.language.data.messages import MessageSeries, Role
from eva.vision.utils import image as image_utils


def format_huggingface_message(
    message: MessageSeries,
    with_images: bool = False,
    image_position: Literal["before_text", "after_text"] = "after_text",
) -> List[Dict[str, Any]]:
    """Formats a message series into a format suitable for Huggingface models.

    Args:
        message: The message series to format.
        with_images: Whether to include images in the formatted message.
        image_position: Position of image relative to text, either "before_text" or "after_text".

    Returns:
        A list of formatted message dictionaries.
    """
    if not with_images:
        return language_utils.format_chat_message(message)

    formatted_message = []
    for item in message:
        if item.role == Role.SYSTEM:
            formatted_message += language_utils.format_chat_message([item])
        else:
            image_content = {"type": "image"}
            text_content = {"type": "text", "text": str(item.content)}

            if image_position == "before_text":
                content = [image_content, text_content]
            elif image_position == "after_text":
                content = [text_content, image_content]
            else:
                raise ValueError(f"Invalid image_position: {image_position}")

            formatted_message.append(
                {
                    "role": item.role,
                    "content": content,
                }
            )
    return formatted_message


def format_litellm_message(
    message: MessageSeries,
    image: tv_tensors.Image | None,
    image_format: Literal["png", "jpeg"] = "jpeg",
    image_position: Literal["before_text", "after_text"] = "after_text",
) -> List[Dict[str, Any]]:
    """Format a message series for LiteLLM API.

    Args:
        message: The message series to format.
        image: Optional image to include in the message.
        image_format: The image format to use for encoding, either "png" or "jpeg".
        image_position: Position of image relative to text, either "before_text" or "after_text".

    Returns:
        A list of formatted message dictionaries.
    """
    if image is None:
        return language_utils.format_chat_message(message)
    image_format = os.getenv("ENCODE_IMAGE_FORMAT", image_format).lower()  # type: ignore

    formatted_message = []
    for item in message:
        if item.role == Role.SYSTEM:
            formatted_message += language_utils.format_chat_message([item])
        else:
            text_content = {"type": "text", "text": str(item.content)}
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": (
                        f"data:image/{image_format};base64,"
                        f"{image_utils.encode_image(image, encoding='base64', file_format=image_format)}"  # noqa: E501
                    )
                },
            }

            if image_position == "before_text":
                content = [image_content, text_content]
            elif image_position == "after_text":
                content = [text_content, image_content]
            else:
                raise ValueError(f"Invalid image_position: {image_position}")

            formatted_message.append(
                {
                    "role": item.role,
                    "content": content,
                }
            )
    return formatted_message
