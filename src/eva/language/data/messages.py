"""Types and classes for conversation messages in a multimodal context."""

import dataclasses
from typing import Any, Dict, List


@dataclasses.dataclass
class Message:
    """Base class for a message in a conversation."""

    content: str
    role: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert the message to a dictionary."""
        return dataclasses.asdict(self)


@dataclasses.dataclass
class UserMessage(Message):
    """User message in a conversation."""

    role: str = "user"


@dataclasses.dataclass
class AssistantMessage(Message):
    """Assistant message in a conversation."""

    role: str = "assistant"


@dataclasses.dataclass
class SystemMessage(Message):
    """System message in a conversation."""

    role: str = "system"


@dataclasses.dataclass
class ModelSystemMessage(SystemMessage):
    """System message for model-specific instructions."""


@dataclasses.dataclass
class TaskSystemMessage(SystemMessage):
    """System message for task-specific instructions."""


MessageSeries = List[Message]
"""A series of conversation messages, can contain a mix of system, user, and AI messages."""
