from __future__ import annotations

from models import RetrievedChunk
from services.generation_service import GenerationService
from services.retrieval_service import RetrievalService
from workflow.graph import build_workflow_graph
from workflow.nodes import citation_node, prompt_selection_node, retrieval_node
from workflow.routing import select_prompt_template
from workflow.state import WorkflowState


class FakeRetrievalService:
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: object | None = None,
        session_id: str | None = None,
        message_id: str | None = None,
        log_retrieval: bool = True,
    ) -> list[RetrievedChunk]:
        return [RetrievedChunk(content=f"Retrieved: {query}", source="document.pdf", score=0.8)]


class FakeGenerationService:
    def __init__(self) -> None:
        self.seen_prompt: str | None = None

    def generate_from_prompt(self, prompt: str) -> str:
        self.seen_prompt = prompt
        return "Generated answer"


def test_select_prompt_template_uses_conversational_when_history_exists() -> None:
    prompt_template = select_prompt_template("User: Hello")

    assert prompt_template.name == "conversational_qa"


def test_prompt_selection_node_builds_prompt_with_query_and_context() -> None:
    state = WorkflowState(query="What is the policy?", session_id=None)
    state.retrieved_chunks = [RetrievedChunk(content="Policy details", source="policy.pdf", score=0.9)]

    updated = prompt_selection_node(state)

    assert updated.prompt is not None
    assert "What is the policy?" in updated.prompt
    assert "Policy details" in updated.prompt
    assert updated.prompt_template_used == "citation_aware_qa"
    assert updated.prompt_version == "v1"


def test_retrieval_node_populates_retrieved_chunks() -> None:
    state = WorkflowState(query="Search this", session_id="abc")
    updated = retrieval_node(state, FakeRetrievalService())

    assert updated.retrieved_chunks == [RetrievedChunk(content="Retrieved: Search this", source="document.pdf", score=0.8)]


def test_full_workflow_graph_executes_sequential_nodes() -> None:
    retrieval_service = FakeRetrievalService()
    generation_service = FakeGenerationService()
    graph = build_workflow_graph(retrieval_service, generation_service)
    state = WorkflowState(query="Explain the process", session_id=None)

    result = graph.run(state)

    assert result.generated_answer is not None
    assert "Generated answer" in result.generated_answer
    assert len(result.citations) == 1
    assert result.evaluation_results["retrieval"].total_count == 1
    assert result.orchestration_metrics is not None
    assert result.orchestration_metrics.retrieved_chunk_count == 1
    assert result.prompt_template_used == "citation_aware_qa"
    assert result.prompt_version == "v1"
    assert generation_service.seen_prompt is not None
