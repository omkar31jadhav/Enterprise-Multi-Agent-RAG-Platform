from datetime import datetime

from database.models import ChatMessageRecord
from models import RetrievedChunk
from rag.memory import SQLConversationMemory
from rag.orchestration import OrchestrationResult, RagOrchestrator


class FakePersistenceService:
    def __init__(self, messages):
        self._messages = messages

    def list_chat_messages(self, session_id: str):
        return [message for message in self._messages if message.session_id == session_id]


def fake_generation_function(prompt: str) -> str:
    return "Generated answer based on prompt."


def test_rag_orchestrator_builds_conversational_prompt_and_formats_citations() -> None:
    messages = [
        ChatMessageRecord(
            message_id="m1",
            session_id="s1",
            role="user",
            content="What is the summary?",
            timestamp=datetime.utcnow(),
        ),
        ChatMessageRecord(
            message_id="m2",
            session_id="s1",
            role="assistant",
            content="This is the start of the conversation.",
            timestamp=datetime.utcnow(),
        ),
    ]
    memory = SQLConversationMemory(
        persistence_service=FakePersistenceService(messages),
        max_history_messages=5,
    )
    orchestrator = RagOrchestrator(memory=memory)
    chunks = [
        RetrievedChunk(
            content="Test content from the document.",
            source="report.pdf",
            score=0.87,
            page_number=12,
            section_title="Executive Summary",
            chunk_id="chunk-123",
            metadata={"topic": "summary"},
        )
    ]

    result = orchestrator.run(
        query="Summarize the key points.",
        retrieved_chunks=chunks,
        session_id="s1",
        message_id="m2",
        generation_function=fake_generation_function,
    )

    assert isinstance(result, OrchestrationResult)
    assert "User: What is the summary?" in result.prompt
    assert "Executive Summary" in result.prompt
    assert result.citations[0].source == "report.pdf"
    assert "Generated answer based on prompt." in result.answer
    assert "References:" in result.answer
    assert result.retrieved_chunks[0].chunk_id == "chunk-123"
    assert result.metrics.prompt_version == "v1"
    assert result.metrics.retrieval_scores == (0.87,)
    assert result.metrics.context_size_estimate > 0
    assert result.metrics.prompt_assembly_time >= 0
    assert result.metrics.generation_time >= 0
    assert result.metrics.total_time >= 0


def test_rag_orchestrator_preserves_retrieval_metadata_in_context() -> None:
    orchestrator = RagOrchestrator(
        memory=SQLConversationMemory(persistence_service=FakePersistenceService([]), max_history_messages=5)
    )
    chunks = [
        RetrievedChunk(
            content="Example content.",
            source="policy.pdf",
            score=0.5,
            page_number=3,
            section_title="Scope",
            chunk_id="policy-001",
            metadata={"policy": True},
        )
    ]

    result = orchestrator.run(
        query="What does the policy cover?",
        retrieved_chunks=chunks,
        session_id="s2",
        generation_function=fake_generation_function,
    )

    assert result.retrieved_chunks[0].source == "policy.pdf"
    assert result.retrieved_chunks[0].page_number == 3
    assert result.retrieved_chunks[0].section_title == "Scope"
    assert result.retrieved_chunks[0].score == 0.5
    assert result.retrieved_chunks[0].chunk_id == "policy-001"
    assert "Scope" in result.prompt
    assert "Source: policy.pdf" in result.prompt


def test_rag_orchestrator_handles_missing_metadata_safely() -> None:
    orchestrator = RagOrchestrator(
        memory=SQLConversationMemory(persistence_service=FakePersistenceService([]), max_history_messages=5)
    )
    chunks = [
        RetrievedChunk(
            content="Content without metadata.",
            source="",
            score=None,
            page_number=None,
            section_title=None,
            chunk_id=None,
            metadata={},
        )
    ]

    result = orchestrator.run(
        query="What is this about?",
        retrieved_chunks=chunks,
        session_id="s3",
        generation_function=fake_generation_function,
    )

    assert "Source: unknown" in result.prompt
    assert "Generated answer based on prompt." in result.answer
    assert result.metrics.retrieved_chunk_count == 1


def test_rag_orchestrator_handles_empty_retrieval_without_crash() -> None:
    orchestrator = RagOrchestrator(
        memory=SQLConversationMemory(persistence_service=FakePersistenceService([]), max_history_messages=5)
    )

    result = orchestrator.run(
        query="No documents are indexed yet.",
        retrieved_chunks=[],
        session_id="s4",
        generation_function=fake_generation_function,
    )

    assert "No relevant document content was found." in result.prompt
    assert result.citations == []
    assert result.metrics.retrieved_chunk_count == 0


def test_rag_orchestrator_falls_back_on_generation_exceptions() -> None:
    def broken_generation(prompt: str) -> str:
        raise RuntimeError("model failure")

    orchestrator = RagOrchestrator(
        memory=SQLConversationMemory(persistence_service=FakePersistenceService([]), max_history_messages=5)
    )

    result = orchestrator.run(
        query="This should fallback.",
        retrieved_chunks=[],
        session_id="s5",
        generation_function=broken_generation,
    )

    assert "The system could not generate an answer at this time." in result.answer
    assert result.citations == []
    assert result.metrics.generation_time >= 0


def test_rag_orchestrator_truncates_long_history() -> None:
    long_messages = [
        ChatMessageRecord(
            message_id=str(i),
            session_id="s6",
            role="user" if i % 2 else "assistant",
            content="A" * 500,
            timestamp=datetime.utcnow(),
        )
        for i in range(15)
    ]
    memory = SQLConversationMemory(
        persistence_service=FakePersistenceService(long_messages),
        max_history_messages=100,
        max_history_chars=800,
    )
    orchestrator = RagOrchestrator(memory=memory)

    result = orchestrator.run(
        query="Check truncation.",
        retrieved_chunks=[],
        session_id="s6",
        generation_function=fake_generation_function,
    )

    assert result.metrics.prompt_name == "conversational_qa"
    assert len(result.prompt) > 0
    assert "..." in result.prompt
    assert "A" in result.prompt
