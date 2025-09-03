"""Tests for message formatting utilities."""

from eva.language.data.messages import (
    AssistantMessage,
    MessageSeries,
    ModelSystemMessage,
    SystemMessage,
    TaskSystemMessage,
    UserMessage,
)
from eva.language.utils.text.messages import (
    batch_insert_system_message,
    combine_system_messages,
    format_chat_message,
    insert_system_message,
    merge_message_contents,
)


def test_format_chat_message_single_message():
    """Test formatting a single message."""
    messages: MessageSeries = [UserMessage(content="Hello")]
    formatted = format_chat_message(messages)

    assert len(formatted) == 1
    assert formatted[0]["role"] == "user"
    assert formatted[0]["content"] == "Hello"


def test_format_chat_message_multiple_messages():
    """Test formatting multiple messages of different types."""
    messages: MessageSeries = [
        SystemMessage(content="System prompt"),
        UserMessage(content="User question"),
        AssistantMessage(content="Assistant response"),
    ]
    formatted = format_chat_message(messages)

    assert len(formatted) == 3
    assert formatted[0] == {"role": "system", "content": "System prompt"}
    assert formatted[1] == {"role": "user", "content": "User question"}
    assert formatted[2] == {"role": "assistant", "content": "Assistant response"}


def test_format_chat_message_empty():
    """Test formatting an empty message series."""
    messages: MessageSeries = []
    formatted = format_chat_message(messages)

    assert formatted == []


def test_combine_system_messages_single_system():
    """Test combining a single system message (should remain unchanged)."""
    messages: MessageSeries = [
        SystemMessage(content="System prompt"),
        UserMessage(content="User question"),
    ]
    combined = combine_system_messages(messages)

    assert len(combined) == 2
    assert combined[0].role == "system"
    assert combined[0].content == "System prompt"
    assert combined[1].role == "user"
    assert combined[1].content == "User question"


def test_combine_system_messages_multiple_system():
    """Test combining multiple system messages."""
    messages: MessageSeries = [
        SystemMessage(content="First system"),
        ModelSystemMessage(content="Model instructions"),
        TaskSystemMessage(content="Task instructions"),
        UserMessage(content="User question"),
    ]
    combined = combine_system_messages(messages)

    assert len(combined) == 2
    assert combined[0].role == "system"
    assert combined[0].content == "First system\nModel instructions\nTask instructions"
    assert combined[1].role == "user"
    assert combined[1].content == "User question"


def test_combine_system_messages_custom_join_char():
    """Test combining system messages with custom join character."""
    messages: MessageSeries = [
        SystemMessage(content="First"),
        SystemMessage(content="Second"),
        UserMessage(content="User"),
    ]
    combined = combine_system_messages(messages, join_char=" | ")

    assert len(combined) == 2
    assert combined[0].content == "First | Second"
    assert combined[1].content == "User"


def test_combine_system_messages_no_system():
    """Test combining when there are no system messages."""
    messages: MessageSeries = [
        UserMessage(content="User question"),
        AssistantMessage(content="Assistant response"),
    ]
    combined = combine_system_messages(messages)

    assert combined == messages


def test_combine_system_messages_only_system():
    """Test combining when there are only system messages."""
    messages: MessageSeries = [
        SystemMessage(content="First"),
        ModelSystemMessage(content="Second"),
        TaskSystemMessage(content="Third"),
    ]
    combined = combine_system_messages(messages)

    assert len(combined) == 1
    assert combined[0].role == "system"
    assert combined[0].content == "First\nSecond\nThird"


def test_merge_message_contents_single():
    """Test merging contents of a single message."""
    messages: MessageSeries = [UserMessage(content="Hello")]
    merged = merge_message_contents(messages)

    assert merged == "Hello"


def test_merge_message_contents_multiple():
    """Test merging contents of multiple messages."""
    messages: MessageSeries = [
        SystemMessage(content="System"),
        UserMessage(content="User"),
        AssistantMessage(content="Assistant"),
    ]
    merged = merge_message_contents(messages)

    assert merged == "System\nUser\nAssistant"


def test_merge_message_contents_custom_join_char():
    """Test merging contents with custom join character."""
    messages: MessageSeries = [
        UserMessage(content="First"),
        UserMessage(content="Second"),
        UserMessage(content="Third"),
    ]
    merged = merge_message_contents(messages, join_char=" -> ")

    assert merged == "First -> Second -> Third"


def test_merge_message_contents_empty():
    """Test merging empty message series."""
    messages: MessageSeries = []
    merged = merge_message_contents(messages)

    assert merged == ""


def test_insert_system_message_with_message():
    """Test inserting a system message."""
    messages: MessageSeries = [
        UserMessage(content="User question"),
        AssistantMessage(content="Response"),
    ]
    system_msg = SystemMessage(content="System prompt")
    result = insert_system_message(messages, system_msg)

    assert len(result) == 3
    assert result[0] == system_msg
    assert result[1].content == "User question"
    assert result[2].content == "Response"


def test_insert_system_message_none():
    """Test inserting None system message (should return original)."""
    messages: MessageSeries = [
        UserMessage(content="User question"),
        AssistantMessage(content="Response"),
    ]
    result = insert_system_message(messages, None)

    assert result == messages


def test_insert_system_message_empty_list():
    """Test inserting system message into empty list."""
    messages: MessageSeries = []
    system_msg = SystemMessage(content="System prompt")
    result = insert_system_message(messages, system_msg)

    assert len(result) == 1
    assert result[0] == system_msg


def test_batch_insert_system_message():
    """Test inserting system message into multiple message series."""
    batch_messages = [
        [UserMessage(content="First user"), AssistantMessage(content="First assistant")],
        [UserMessage(content="Second user")],
        [],
    ]
    system_msg = SystemMessage(content="System prompt")
    result = batch_insert_system_message(batch_messages, system_msg)

    assert len(result) == 3
    assert len(result[0]) == 3
    assert result[0][0] == system_msg
    assert result[0][1].content == "First user"
    assert result[0][2].content == "First assistant"

    assert len(result[1]) == 2
    assert result[1][0] == system_msg
    assert result[1][1].content == "Second user"

    assert len(result[2]) == 1
    assert result[2][0] == system_msg


def test_batch_insert_system_message_none():
    """Test batch inserting None system message."""
    batch_messages = [
        [UserMessage(content="First user")],
        [UserMessage(content="Second user")],
    ]
    result = batch_insert_system_message(batch_messages, None)  # type: ignore

    assert result == batch_messages


def test_batch_insert_system_message_empty_batch():
    """Test batch inserting into empty batch."""
    batch_messages = []
    system_msg = SystemMessage(content="System prompt")
    result = batch_insert_system_message(batch_messages, system_msg)

    assert result == []
