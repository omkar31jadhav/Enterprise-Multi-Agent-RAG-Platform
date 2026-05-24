from datetime import datetime

from database.connection import Database
from database.models import ChatMessageRecord
from database.services import PersistenceService
from models import DocumentChunk, RetrievedChunk
from rag.memory import SQLConversationMemory
from rag.orchestration import OrchestrationResult, RagOrchestrator
from services.rag_service import RagService
from services.retrieval_service import RetrievalService
from vectorstores.base import BaseVectorStore, CollectionStats, MetadataFilter


class FakePersistenceService:
    def __init__(self, messages=None):
        self.messages = messages or []

    def add_message(self, session_id, role, content, metadata=None, message_id=None):
        message = ChatMessageRecord(
            message_id=message_id or f"msg_{len(self.messages)+1}",
            session_id=session_id,
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
        )
        self.messages.append(message)
        return message

    def list_chat_messages(self, session_id: str):
        return [message for message in self.messages if message.session_id == session_id]


class FakeRetrievalService:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, query: str, top_k=None, filters=None, session_id=None, message_id=None, log_retrieval=True):
        return self.chunks


class FakeGenerationService:
    def __init__(self):
        self.last_prompt: str | None = None

    def generate_from_prompt(self, prompt: str) -> str:
        self.last_prompt = prompt
        return "Generated answer."


class InMemoryVectorStore(BaseVectorStore):
    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self._chunks = chunks or []

    def clear(self) -> None:
        self._chunks.clear()

    def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        return 0

    def similarity_search(
        self,
        query: str,
        k: int,
        filters: MetadataFilter | None = None,
    ) -> list[RetrievedChunk]:
        return self._chunks[:k]

    def collection_stats(self) -> CollectionStats:
        return CollectionStats(
            backend="in-memory",
            collection_name="test",
            record_count=len(self._chunks),
            persist_path=None,
        )

    @property
    def records(self) -> list[dict[str, object]]:
        return [
            {
                "chunk": chunk.content,
                "metadata": chunk.metadata,
                "source": chunk.source,
            }
            for chunk in self._chunks
        ]


class CapturingOrchestrator:
    def __init__(self, orchestrator: RagOrchestrator) -> None:
        self._orchestrator = orchestrator
        self.last_result: OrchestrationResult | None = None

    def run(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        session_id: str | None = None,
        message_id: str | None = None,
        generation_function=None,
    ) -> OrchestrationResult:
        self.last_result = self._orchestrator.run(
            query=query,
            retrieved_chunks=retrieved_chunks,
            session_id=session_id,
            message_id=message_id,
            generation_function=generation_function,
        )
        return self.last_result


def test_rag_service_answer_question_uses_orchestrator_and_returns_citation_references() -> None:
    chunks = [
        RetrievedChunk(
            content="Context text.",
            source="doc.pdf",
            score=0.8,
            page_number=10,
            section_title="Introduction",
            chunk_id="chunk-1",
            metadata={"topic": "test"},
        )
    ]
    retrieval_service = FakeRetrievalService(chunks=chunks)
    generation_service = FakeGenerationService()
    service = RagService(
        retrieval_service=retrieval_service,
        generation_service=generation_service,
        persistence_service=None,
        orchestrator=RagOrchestrator(),
    )

    answer = service.answer_question("What is the summary?")

    assert answer.startswith("Generated answer.")
    assert "References:" in answer
    assert "Source: doc.pdf" in answer
    assert generation_service.last_prompt is not None
    assert "Introduction" in generation_service.last_prompt
    assert "What is the summary?" in generation_service.last_prompt


def test_rag_service_answer_chat_uses_memory_and_persists_assistant_message() -> None:
    messages = [
        ChatMessageRecord(
            message_id="m1",
            session_id="session-1",
            role="user",
            content="Hello",
            timestamp=datetime.utcnow(),
        )
    ]
    persistence = FakePersistenceService(messages=messages)
    memory = SQLConversationMemory(persistence_service=persistence, max_history_messages=5)
    orchestrator = RagOrchestrator(memory=memory)
    chunks = [
        RetrievedChunk(
            content="Chat context.",
            source="session.pdf",
            score=0.9,
            page_number=2,
            section_title="Overview",
            chunk_id="chunk-2",
            metadata={"chat": True},
        )
    ]
    retrieval_service = FakeRetrievalService(chunks=chunks)
    generation_service = FakeGenerationService()
    service = RagService(
        retrieval_service=retrieval_service,
        generation_service=generation_service,
        persistence_service=persistence,
        orchestrator=orchestrator,
    )

    answer, user_message, assistant_message = service.answer_chat(
        "How does this work?",
        session_id="session-1",
    )

    assert answer.startswith("Generated answer.")
    assert "References:" in answer
    assert user_message.role == "user"
    assert assistant_message.role == "assistant"
    assert assistant_message.metadata["retrieved_chunk_ids"] == ["chunk-2"]
    assert any("User: Hello" in m for m in generation_service.last_prompt.splitlines())


def test_rag_service_answer_question_handles_empty_retrieval_gracefully() -> None:
    retrieval_service = FakeRetrievalService(chunks=[])
    generation_service = FakeGenerationService()
    service = RagService(
        retrieval_service=retrieval_service,
        generation_service=generation_service,
        persistence_service=None,
        orchestrator=RagOrchestrator(),
    )

    answer = service.answer_question("What is the summary?")

    assert answer.startswith("Generated answer.")
    assert "References:" not in answer or answer.endswith("References:\n") is False
    assert generation_service.last_prompt is not None


def test_rag_service_answer_question_full_flow_records_retrieval_and_citation_metadata(tmp_path, caplog) -> None:
    database = Database(sqlite_path=tmp_path / "question_integration.db")
    persistence = PersistenceService(database=database, initialize=True)
    chunks = [
        RetrievedChunk(
            content="Complete context for the question.",
            source="doc.pdf",
            score=0.95,
            page_number=5,
            section_title="Details",
            chunk_id="chunk-full-1",
            metadata={"topic": "integration"},
        )
    ]
    retrieval_service = RetrievalService(
        vector_store=InMemoryVectorStore(chunks=chunks),
        persistence_service=persistence,
    )
    generation_service = FakeGenerationService()
    orchestrator = CapturingOrchestrator(RagOrchestrator())
    service = RagService(
        retrieval_service=retrieval_service,
        generation_service=generation_service,
        persistence_service=persistence,
        orchestrator=orchestrator,
    )

    with caplog.at_level("INFO"):
        answer = service.answer_question("What are the key details?")

    assert answer.startswith("Generated answer.")
    assert "References:" in answer
    assert "Source: doc.pdf" in answer
    assert "Page: 5" in generation_service.last_prompt
    assert "Section: Details" in generation_service.last_prompt
    assert "Score: 0.9500" in generation_service.last_prompt
    assert orchestrator.last_result is not None
    assert orchestrator.last_result.metrics.retrieved_chunk_count == 1
    assert orchestrator.last_result.metrics.citation_count == 1
    assert orchestrator.last_result.metrics.generation_time >= 0
    assert "retrieval_time=" in caplog.text

    retrieval_logs = persistence.list_recent_retrievals(limit=5)
    assert len(retrieval_logs) == 1
    assert retrieval_logs[0].document_name == "doc.pdf"
    assert retrieval_logs[0].page_number == 5
    assert retrieval_logs[0].section_title == "Details"
    assert abs(retrieval_logs[0].similarity_score - 0.95) < 1e-6


def test_rag_service_answer_chat_full_flow_uses_conversational_memory(tmp_path) -> None:
    database = Database(sqlite_path=tmp_path / "chat_integration.db")
    persistence = PersistenceService(database=database, initialize=True)
    persistence.create_chat_session(session_id="session-123")
    persistence.add_message(
        session_id="session-123",
        role="user",
        content="Hello, please remember this thread.",
    )
    chunks = [
        RetrievedChunk(
            content="Chat conversation context.",
            source="chat.pdf",
            score=0.87,
            page_number=1,
            section_title="Overview",
            chunk_id="chunk-chat-1",
            metadata={"chat": True},
        )
    ]
    retrieval_service = RetrievalService(
        vector_store=InMemoryVectorStore(chunks=chunks),
        persistence_service=persistence,
    )
    generation_service = FakeGenerationService()
    memory = SQLConversationMemory(persistence_service=persistence, max_history_messages=5)
    orchestrator = CapturingOrchestrator(RagOrchestrator(memory=memory))
    service = RagService(
        retrieval_service=retrieval_service,
        generation_service=generation_service,
        persistence_service=persistence,
        orchestrator=orchestrator,
    )

    answer, user_message, assistant_message = service.answer_chat(
        "What should I do next?",
        session_id="session-123",
    )

    assert answer.startswith("Generated answer.")
    assert "References:" in answer
    assert "User: Hello, please remember this thread." in generation_service.last_prompt
    assert "User: What should I do next?" in generation_service.last_prompt
    assert assistant_message.role == "assistant"
    assert assistant_message.content == answer
    assert assistant_message.metadata["retrieved_chunk_ids"] == ["chunk-chat-1"]
    assert orchestrator.last_result is not None
    assert orchestrator.last_result.metrics.prompt_name == "conversational_qa"

    persisted_messages = persistence.list_chat_messages("session-123")
    assert len(persisted_messages) == 3
    assert persisted_messages[-1].role == "assistant"
    retrieval_logs = persistence.list_recent_retrievals(limit=5, session_id="session-123")
    assert len(retrieval_logs) == 1


def test_rag_service_answer_question_fallbacks_on_generation_exception(tmp_path) -> None:
    database = Database(sqlite_path=tmp_path / "fallback_integration.db")
    persistence = PersistenceService(database=database, initialize=True)
    retrieval_service = RetrievalService(
        vector_store=InMemoryVectorStore(chunks=[]),
        persistence_service=persistence,
    )

    class BrokenGenerationService:
        def generate_from_prompt(self, prompt: str) -> str:
            raise RuntimeError("model unavailable")

    generation_service = BrokenGenerationService()
    orchestrator = CapturingOrchestrator(RagOrchestrator())
    service = RagService(
        retrieval_service=retrieval_service,
        generation_service=generation_service,
        persistence_service=persistence,
        orchestrator=orchestrator,
    )

    answer = service.answer_question("Is anything available?")

    assert "could not generate an answer" in answer
    assert "References:" not in answer
    assert orchestrator.last_result is not None
    assert orchestrator.last_result.metrics.retrieved_chunk_count == 0
    assert orchestrator.last_result.metrics.citation_count == 0
    assert persistence.list_recent_retrievals(limit=5) == []


def test_rag_service_answer_question_handles_missing_retrieval_metadata_gracefully(tmp_path) -> None:
    database = Database(sqlite_path=tmp_path / "missing_metadata.db")
    persistence = PersistenceService(database=database, initialize=True)
    chunks = [
        RetrievedChunk(
            content="Metadata-less content.",
            source="",
            score=None,
            page_number=None,
            section_title=None,
            chunk_id=None,
            metadata={},
        )
    ]
    retrieval_service = RetrievalService(
        vector_store=InMemoryVectorStore(chunks=chunks),
        persistence_service=persistence,
    )
    generation_service = FakeGenerationService()
    orchestrator = CapturingOrchestrator(RagOrchestrator())
    service = RagService(
        retrieval_service=retrieval_service,
        generation_service=generation_service,
        persistence_service=persistence,
        orchestrator=orchestrator,
    )

    answer = service.answer_question("Check missing metadata.")

    assert answer.startswith("Generated answer.")
    assert "References:" in answer
    assert "Source: unknown" in generation_service.last_prompt
    assert orchestrator.last_result is not None
    assert orchestrator.last_result.metrics.retrieved_chunk_count == 1
    assert orchestrator.last_result.metrics.citation_count == 1


def test_rag_service_answer_question_with_no_history_uses_citation_aware_prompt(tmp_path) -> None:
    database = Database(sqlite_path=tmp_path / "no_history.db")
    persistence = PersistenceService(database=database, initialize=True)
    chunks = [
        RetrievedChunk(
            content="Fresh context without prior history.",
            source="fresh.pdf",
            score=0.75,
            page_number=7,
            section_title="Fresh",
            chunk_id="chunk-fresh-1",
            metadata={"topic": "none"},
        )
    ]
    retrieval_service = RetrievalService(
        vector_store=InMemoryVectorStore(chunks=chunks),
        persistence_service=persistence,
    )
    generation_service = FakeGenerationService()
    orchestrator = CapturingOrchestrator(RagOrchestrator())
    service = RagService(
        retrieval_service=retrieval_service,
        generation_service=generation_service,
        persistence_service=persistence,
        orchestrator=orchestrator,
    )

    answer = service.answer_question("What is fresh?")

    assert answer.startswith("Generated answer.")
    assert orchestrator.last_result is not None
    assert orchestrator.last_result.metrics.prompt_name == "citation_aware_qa"
    assert orchestrator.last_result.metrics.history_count == 0
    assert "User:" not in generation_service.last_prompt
    assert "What is fresh?" in generation_service.last_prompt
