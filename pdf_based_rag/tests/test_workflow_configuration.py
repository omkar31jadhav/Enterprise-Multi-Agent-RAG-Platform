from datetime import datetime
from typing import Any

from config.workflow import WorkflowConfig
from database.connection import Database
from database.models import ChatMessageRecord
from database.services import PersistenceService
from models import DocumentChunk, RetrievedChunk
from rag.memory import SQLConversationMemory
from rag.orchestration import RagOrchestrator
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


class CapturingTrackingAdapter:
    def __init__(self) -> None:
        self.params: dict[str, Any] = {}
        self.metrics: dict[str, float | int] = {}

    def start_run(self, config=None) -> None:
        pass

    def log_params(self, params: dict[str, Any]) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict[str, float | int]) -> None:
        self.metrics.update(metrics)

    def set_tags(self, tags: dict[str, str]) -> None:
        pass

    def end_run(self) -> None:
        pass


def test_rag_service_with_workflow_disabled_uses_classic_path() -> None:
    chunks = [
        RetrievedChunk(
            content="Context.",
            source="doc.pdf",
            score=0.85,
            chunk_id="chunk-1",
        )
    ]
    retrieval_service = FakeRetrievalService(chunks=chunks)
    generation_service = FakeGenerationService()
    tracking_adapter = CapturingTrackingAdapter()
    config = WorkflowConfig.disabled()
    service = RagService(
        retrieval_service=retrieval_service,
        generation_service=generation_service,
        persistence_service=FakePersistenceService(),
        tracking_adapter=tracking_adapter,
        workflow_config=config,
    )

    answer = service.answer_question("What is this?")

    assert answer.startswith("Generated answer.")
    assert tracking_adapter.params["orchestration_path"] == "classic"
    assert "workflow" not in tracking_adapter.params.get("execution_path", "")


def test_rag_service_with_workflow_enabled_uses_workflow_path() -> None:
    chunks = [
        RetrievedChunk(
            content="Workflow context.",
            source="doc.pdf",
            score=0.85,
            chunk_id="chunk-1",
        )
    ]
    retrieval_service = FakeRetrievalService(chunks=chunks)
    generation_service = FakeGenerationService()
    tracking_adapter = CapturingTrackingAdapter()
    config = WorkflowConfig.enabled_with_fallback()
    service = RagService(
        retrieval_service=retrieval_service,
        generation_service=generation_service,
        persistence_service=FakePersistenceService(),
        tracking_adapter=tracking_adapter,
        workflow_config=config,
    )

    answer = service.answer_question("What is this?")

    assert answer.startswith("Generated answer.")
    assert "workflow_execution_id" in tracking_adapter.params
    assert tracking_adapter.metrics["workflow_duration"] >= 0


def test_answer_question_use_workflow_parameter_overrides_config() -> None:
    chunks = [
        RetrievedChunk(
            content="Context.",
            source="doc.pdf",
            score=0.85,
            chunk_id="chunk-1",
        )
    ]
    retrieval_service = FakeRetrievalService(chunks=chunks)
    generation_service = FakeGenerationService()
    tracking_adapter = CapturingTrackingAdapter()
    config = WorkflowConfig.disabled()  # Disabled by default
    service = RagService(
        retrieval_service=retrieval_service,
        generation_service=generation_service,
        persistence_service=FakePersistenceService(),
        tracking_adapter=tracking_adapter,
        workflow_config=config,
    )

    # Override config to use workflow
    answer = service.answer_question("What is this?", use_workflow=True)

    assert "workflow_execution_id" in tracking_adapter.params


def test_answer_chat_use_workflow_parameter_overrides_config() -> None:
    chunks = [
        RetrievedChunk(
            content="Chat context.",
            source="chat.pdf",
            score=0.85,
            chunk_id="chat-1",
        )
    ]
    persistence = FakePersistenceService()
    retrieval_service = FakeRetrievalService(chunks=chunks)
    generation_service = FakeGenerationService()
    tracking_adapter = CapturingTrackingAdapter()
    config = WorkflowConfig.disabled()  # Disabled by default
    service = RagService(
        retrieval_service=retrieval_service,
        generation_service=generation_service,
        persistence_service=persistence,
        tracking_adapter=tracking_adapter,
        workflow_config=config,
    )

    # Override config to use workflow
    answer, user_msg, asst_msg = service.answer_chat("Question?", session_id="session-1", use_workflow=True)

    assert "workflow_execution_id" in tracking_adapter.params
    assert user_msg.role == "user"
    assert asst_msg.role == "assistant"


def test_rag_service_falls_back_to_classic_on_workflow_failure() -> None:
    class FailingGenerationService:
        def generate_from_prompt(self, prompt: str) -> str:
            raise RuntimeError("generation failure")

    chunks = [
        RetrievedChunk(
            content="Context.",
            source="doc.pdf",
            score=0.85,
            chunk_id="chunk-1",
        )
    ]
    retrieval_service = FakeRetrievalService(chunks=chunks)
    generation_service = FailingGenerationService()
    tracking_adapter = CapturingTrackingAdapter()
    config = WorkflowConfig.enabled_with_fallback()
    service = RagService(
        retrieval_service=retrieval_service,
        generation_service=generation_service,
        persistence_service=FakePersistenceService(),
        tracking_adapter=tracking_adapter,
        workflow_config=config,
    )

    answer = service.answer_question("What is this?")

    # Workflow execution detects node failure and falls back to classic path
    # Classic path also fails due to failing generation service, returns orchestrator fallback
    assert "could not generate an answer" in answer or "could not complete" in answer
    assert tracking_adapter.metrics.get("workflow_fallback_count") == 1


def test_rag_service_raises_on_workflow_failure_without_fallback() -> None:
    class FailingGenerationService:
        def generate_from_prompt(self, prompt: str) -> str:
            raise RuntimeError("generation failure")

    chunks = [
        RetrievedChunk(
            content="Context.",
            source="doc.pdf",
            score=0.85,
            chunk_id="chunk-1",
        )
    ]
    retrieval_service = FakeRetrievalService(chunks=chunks)
    generation_service = FailingGenerationService()
    tracking_adapter = CapturingTrackingAdapter()
    config = WorkflowConfig(
        workflow_enabled=True,
        workflow_fallback_enabled=False,  # No fallback
    )
    service = RagService(
        retrieval_service=retrieval_service,
        generation_service=generation_service,
        persistence_service=FakePersistenceService(),
        tracking_adapter=tracking_adapter,
        workflow_config=config,
    )

    try:
        service.answer_question("What is this?")
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass


def test_workflow_config_from_env() -> None:
    import os

    os.environ["WORKFLOW_ENABLED"] = "true"
    os.environ["WORKFLOW_FALLBACK_ENABLED"] = "true"

    config = WorkflowConfig.from_env()

    assert config.workflow_enabled is True
    assert config.workflow_fallback_enabled is True


def test_workflow_evaluation_disabled_skips_evaluation() -> None:
    chunks = [
        RetrievedChunk(
            content="Context.",
            source="doc.pdf",
            score=0.85,
            chunk_id="chunk-1",
        )
    ]
    retrieval_service = FakeRetrievalService(chunks=chunks)
    generation_service = FakeGenerationService()
    tracking_adapter = CapturingTrackingAdapter()
    config = WorkflowConfig(
        workflow_enabled=True,
        workflow_fallback_enabled=True,
        workflow_evaluation_enabled=False,
    )
    service = RagService(
        retrieval_service=retrieval_service,
        generation_service=generation_service,
        persistence_service=FakePersistenceService(),
        tracking_adapter=tracking_adapter,
        workflow_config=config,
    )

    answer = service.answer_question("What is this?")

    assert answer.startswith("Generated answer.")
