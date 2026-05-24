from __future__ import annotations

from collections.abc import Sequence

from database.models import ChatMessageRecord
from database.services import PersistenceService
from utils import get_logger

logger = get_logger(__name__)


class SQLConversationMemory:
    def __init__(
        self,
        persistence_service: PersistenceService | None = None,
        max_history_messages: int = 20,
    ) -> None:
        self.persistence_service = persistence_service or PersistenceService()
        self.max_history_messages = max_history_messages

    def load_history(self, session_id: str) -> list[ChatMessageRecord]:
        logger.info("Loading conversation memory for session_id=%s max_history=%s", session_id, self.max_history_messages)
        messages = self.persistence_service.list_chat_messages(session_id=session_id)
        if len(messages) > self.max_history_messages:
            return messages[-self.max_history_messages :]
        return messages

    def format_history(self, session_id: str) -> str:
        messages = self.load_history(session_id)
        formatted = [f"{message.role.title()}: {message.content}" for message in messages]
        return "\n".join(formatted)
