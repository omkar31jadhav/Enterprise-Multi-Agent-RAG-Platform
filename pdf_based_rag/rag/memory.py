from __future__ import annotations

from collections.abc import Sequence

from database.models import ChatMessageRecord
from database.services import PersistenceService
from utils import get_logger

logger = get_logger(__name__)


class SQLConversationMemory:
    """SQL-backed memory for prompt-ready conversational history."""

    def __init__(
        self,
        persistence_service: PersistenceService | None = None,
        max_history_messages: int = 20,
        max_history_chars: int = 3000,
    ) -> None:
        self.persistence_service = persistence_service or PersistenceService()
        self.max_history_messages = max_history_messages
        self.max_history_chars = max_history_chars

    def load_history(self, session_id: str) -> list[ChatMessageRecord]:
        logger.info(
            "Loading conversation memory for session_id=%s max_history_messages=%s",
            session_id,
            self.max_history_messages,
        )
        messages = self.persistence_service.list_chat_messages(session_id=session_id)
        if len(messages) > self.max_history_messages:
            return messages[-self.max_history_messages :]
        return messages

    def format_history(self, session_id: str) -> str:
        try:
            history = self.load_history(session_id)
        except Exception as exc:
            logger.exception(
                "Failed to load conversation history for session_id=%s", session_id
            )
            return ""

        if not history:
            return ""

        formatted_messages: list[str] = []
        for message in history:
            role = message.role.strip().title() or "User"
            content = message.content.strip()
            if not content:
                continue
            formatted_messages.append(f"{role}: {content}")

        return "\n".join(formatted_messages)

    def prompt_ready_history(self, session_id: str) -> str:
        formatted_history = self.format_history(session_id)
        if not formatted_history:
            return ""
        return self._truncate_history(formatted_history)

    def _truncate_history(self, history: str) -> str:
        if len(history) <= self.max_history_chars:
            return history
        truncated = history[-self.max_history_chars :].lstrip()
        logger.warning(
            "Truncated conversation history from %s to %s characters",
            len(history),
            len(truncated),
        )
        return f"...{truncated}"
