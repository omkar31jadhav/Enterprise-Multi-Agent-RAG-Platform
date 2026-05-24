from datetime import datetime, timedelta

from database.models import ChatMessageRecord
from rag.memory import SQLConversationMemory


class FakePersistenceService:
    def __init__(self, messages):
        self._messages = messages

    def list_chat_messages(self, session_id: str):
        return [message for message in self._messages if message.session_id == session_id]


def test_sql_conversation_memory_formats_history_roles() -> None:
    messages = [
        ChatMessageRecord(
            message_id="m1",
            session_id="s1",
            role="user",
            content="Hello",
            timestamp=datetime.utcnow(),
        ),
        ChatMessageRecord(
            message_id="m2",
            session_id="s1",
            role="assistant",
            content="Hi there.",
            timestamp=datetime.utcnow(),
        ),
    ]
    memory = SQLConversationMemory(
        persistence_service=FakePersistenceService(messages),
        max_history_messages=10,
    )

    formatted = memory.format_history("s1")

    assert formatted == "User: Hello\nAssistant: Hi there."


def test_sql_conversation_memory_truncates_to_max_history_messages() -> None:
    now = datetime.utcnow()
    messages = [
        ChatMessageRecord(
            message_id=f"m{i}",
            session_id="s1",
            role="user" if i % 2 else "assistant",
            content=f"message {i}",
            timestamp=now + timedelta(seconds=i),
        )
        for i in range(1, 6)
    ]

    memory = SQLConversationMemory(
        persistence_service=FakePersistenceService(messages),
        max_history_messages=3,
    )

    formatted = memory.format_history("s1")

    assert "message 3" in formatted
    assert "message 4" in formatted
    assert "message 5" in formatted
    assert "message 2" not in formatted


def test_sql_conversation_memory_returns_empty_for_no_history() -> None:
    memory = SQLConversationMemory(
        persistence_service=FakePersistenceService([]),
        max_history_messages=5,
    )

    result = memory.prompt_ready_history("empty-session")

    assert result == ""


def test_sql_conversation_memory_truncates_long_history() -> None:
    now = datetime.utcnow()
    messages = [
        ChatMessageRecord(
            message_id=f"m{i}",
            session_id="s1",
            role="user" if i % 2 else "assistant",
            content="A" * 200,
            timestamp=now + timedelta(seconds=i),
        )
        for i in range(20)
    ]

    memory = SQLConversationMemory(
        persistence_service=FakePersistenceService(messages),
        max_history_messages=20,
        max_history_chars=500,
    )

    result = memory.prompt_ready_history("s1")

    assert len(result) <= 504
    assert result.startswith("...")
    assert "A" in result
