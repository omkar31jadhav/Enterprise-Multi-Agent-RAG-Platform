from __future__ import annotations

import time
from typing import TYPE_CHECKING
from uuid import uuid4

from config.workflow import WorkflowConfig
from database.models import ChatMessageRecord
from database.services import PersistenceService
from rag.orchestration import RagOrchestrator
from services.generation_service import GenerationService
from services.retrieval_service import RetrievalService
from utils import get_logger

if TYPE_CHECKING:
    from ml.tracking import TrackingAdapter
    from workflow.graph import WorkflowGraph
    from workflow.state import WorkflowState


logger = get_logger(__name__)


class RagService:
    def __init__(
        self,
        retrieval_service: RetrievalService | None = None,
        generation_service: GenerationService | None = None,
        persistence_service: PersistenceService | None = None,
        orchestrator: RagOrchestrator | None = None,
        tracking_adapter: TrackingAdapter | None = None,
        workflow_graph: WorkflowGraph | None = None,
        workflow_config: WorkflowConfig | None = None,
    ) -> None:
        self.persistence_service = persistence_service
        self.workflow_config = workflow_config or WorkflowConfig.disabled()
        if retrieval_service is None:
            self.persistence_service = self.persistence_service or PersistenceService()
            retrieval_service = RetrievalService(persistence_service=self.persistence_service)
        self.retrieval_service = retrieval_service
        self.generation_service = generation_service or GenerationService()
        self.tracking_adapter: TrackingAdapter
        if tracking_adapter is None:
            from ml.tracking import NoOpTrackingAdapter

            self.tracking_adapter = NoOpTrackingAdapter()
        else:
            self.tracking_adapter = tracking_adapter
        self.orchestrator = orchestrator or RagOrchestrator(tracking_adapter=self.tracking_adapter)
        if workflow_graph is None:
            from workflow.graph import build_workflow_graph

            orchestrator_memory = getattr(self.orchestrator, "memory", None)
            self.workflow_graph = build_workflow_graph(
                self.retrieval_service,
                self.generation_service,
                orchestrator_memory,
                self.workflow_config,
            )
        else:
            self.workflow_graph = workflow_graph

    def answer_question(self, query: str, use_workflow: bool | None = None) -> str:
        if use_workflow is None:
            use_workflow = self.workflow_config.workflow_enabled

        if use_workflow:
            return self._answer_question_with_workflow_and_fallback(query)
        return self._answer_question_with_orchestration(query)

    def _answer_question_with_orchestration(self, query: str) -> str:
        retrieval_start = time.perf_counter()
        retrieved_chunks = self.retrieval_service.retrieve(query)
        retrieval_time = time.perf_counter() - retrieval_start
        logger.info(
            "Answering query with %s retrieved chunks retrieval_time=%.4f orchestration_path=classic",
            len(retrieved_chunks),
            retrieval_time,
        )
        self.tracking_adapter.log_metrics({"retrieval_time": retrieval_time})
        self.tracking_adapter.log_params(
            {
                "query": query,
                "execution_path": "answer_question",
                "orchestration_path": "classic",
                "session_id": "none",
            }
        )

        orchestration_result = self.orchestrator.run(
            query=query,
            retrieved_chunks=retrieved_chunks,
            session_id=None,
            message_id=None,
            generation_function=self._generate_from_prompt,
        )
        self.tracking_adapter.log_params(
            {
                "prompt_name": orchestration_result.metrics.prompt_name,
                "prompt_version": orchestration_result.metrics.prompt_version,
                "prompt_template_used": orchestration_result.metrics.prompt_template_used,
                "orchestration_path": "classic",
            }
        )
        self.tracking_adapter.log_metrics(
            {
                "generation_time": orchestration_result.metrics.generation_time,
                "orchestration_total_time": orchestration_result.metrics.total_time,
                "retrieved_chunk_count": orchestration_result.metrics.retrieved_chunk_count,
                "citation_count": orchestration_result.metrics.citation_count,
                "history_message_count": orchestration_result.metrics.history_count,
                "context_size_estimate": orchestration_result.metrics.context_size_estimate,
            }
        )
        logger.info(
            "Orchestration completed for query=%s prompt_name=%s citations=%s prompt_time=%.4f generation_time=%.4f total_time=%.4f orchestration_path=classic",
            query,
            orchestration_result.metrics.prompt_name,
            orchestration_result.metrics.citation_count,
            orchestration_result.metrics.prompt_assembly_time,
            orchestration_result.metrics.generation_time,
            orchestration_result.metrics.total_time,
        )
        return orchestration_result.answer

    def _answer_question_with_workflow_and_fallback(self, query: str) -> str:
        workflow_id = str(uuid4())
        from workflow.state import WorkflowState

        state = WorkflowState(query=query, workflow_execution_id=workflow_id, workflow_path="answer_question_with_workflow")
        try:
            response_state = self._run_workflow(state)
            # Check if workflow had node failures
            if response_state.node_failures:
                failure_msg = "; ".join(f"{node}: {error}" for node, error in response_state.node_failures.items())
                if not self.workflow_config.workflow_fallback_enabled:
                    raise RuntimeError(f"Workflow execution failed with node failures: {failure_msg}")
                logger.warning(
                    "Workflow execution had node failures for query=%s: %s. Falling back to classic orchestration.",
                    query,
                    failure_msg,
                )
                self.tracking_adapter.log_params({"orchestration_fallback": "classic"})
                self.tracking_adapter.log_metrics({"workflow_fallback_count": 1})
                return self._answer_question_with_orchestration(query)
            answer = response_state.generated_answer or "The workflow did not produce an answer."
            return answer
        except Exception as exc:
            if self.workflow_config.workflow_fallback_enabled:
                logger.warning(
                    "Workflow execution failed for query=%s with error=%s. Falling back to classic orchestration.",
                    query,
                    str(exc),
                )
                self.tracking_adapter.log_params({"orchestration_fallback": "classic"})
                self.tracking_adapter.log_metrics({"workflow_fallback_count": 1})
                return self._answer_question_with_orchestration(query)
            else:
                raise

    def answer_chat(
        self,
        query: str,
        session_id: str,
        use_workflow: bool | None = None,
    ) -> tuple[str, ChatMessageRecord, ChatMessageRecord]:
        if use_workflow is None:
            use_workflow = self.workflow_config.workflow_enabled

        if use_workflow:
            return self._answer_chat_with_workflow_and_fallback(query, session_id)
        return self._answer_chat_with_orchestration(query, session_id)

    def _answer_chat_with_orchestration(self, query: str, session_id: str) -> tuple[str, ChatMessageRecord, ChatMessageRecord]:
        if self.persistence_service is None:
            self.persistence_service = PersistenceService()
        user_message = self.persistence_service.add_message(
            session_id=session_id,
            role="user",
            content=query,
        )
        retrieval_start = time.perf_counter()
        retrieved_chunks = self.retrieval_service.retrieve(
            query,
            session_id=session_id,
            message_id=user_message.message_id,
        )
        retrieval_time = time.perf_counter() - retrieval_start
        logger.info(
            "Answering chat query with %s retrieved chunks retrieval_time=%.4f session_id=%s orchestration_path=classic",
            len(retrieved_chunks),
            retrieval_time,
            session_id,
        )
        self.tracking_adapter.log_metrics({"retrieval_time": retrieval_time})
        self.tracking_adapter.log_params(
            {
                "query": query,
                "execution_path": "answer_chat",
                "orchestration_path": "classic",
                "session_id": session_id,
                "message_id": user_message.message_id,
            }
        )

        orchestration_result = self.orchestrator.run(
            query=query,
            retrieved_chunks=retrieved_chunks,
            session_id=session_id,
            message_id=user_message.message_id,
            generation_function=self._generate_from_prompt,
        )

        answer = orchestration_result.answer
        self.tracking_adapter.log_params(
            {
                "prompt_name": orchestration_result.metrics.prompt_name,
                "prompt_version": orchestration_result.metrics.prompt_version,
                "prompt_template_used": orchestration_result.metrics.prompt_template_used,
                "orchestration_path": "classic",
            }
        )
        self.tracking_adapter.log_metrics(
            {
                "generation_time": orchestration_result.metrics.generation_time,
                "orchestration_total_time": orchestration_result.metrics.total_time,
                "retrieved_chunk_count": orchestration_result.metrics.retrieved_chunk_count,
                "citation_count": orchestration_result.metrics.citation_count,
                "history_message_count": orchestration_result.metrics.history_count,
                "context_size_estimate": orchestration_result.metrics.context_size_estimate,
            }
        )
        assistant_message = self.persistence_service.add_message(
            session_id=session_id,
            role="assistant",
            content=answer,
            metadata={
                "retrieved_chunk_ids": [
                    chunk.chunk_id for chunk in retrieved_chunks if chunk.chunk_id
                ]
            },
        )

        logger.info(
            "Orchestration completed for chat session_id=%s message_id=%s prompt_name=%s citations=%s total_time=%.4f orchestration_path=classic",
            session_id,
            user_message.message_id,
            orchestration_result.metrics.prompt_name,
            orchestration_result.metrics.citation_count,
            orchestration_result.metrics.total_time,
        )
        return answer, user_message, assistant_message

    def _answer_chat_with_workflow_and_fallback(
        self,
        query: str,
        session_id: str,
    ) -> tuple[str, ChatMessageRecord, ChatMessageRecord]:
        if self.persistence_service is None:
            self.persistence_service = PersistenceService()
        user_message = self.persistence_service.add_message(
            session_id=session_id,
            role="user",
            content=query,
        )
        workflow_id = str(uuid4())
        from workflow.state import WorkflowState

        state = WorkflowState(
            query=query,
            session_id=session_id,
            message_id=user_message.message_id,
            workflow_execution_id=workflow_id,
            workflow_path="answer_chat_with_workflow",
        )
        try:
            response_state = self._run_workflow(state)
            # Check if workflow had node failures
            if response_state.node_failures:
                failure_msg = "; ".join(f"{node}: {error}" for node, error in response_state.node_failures.items())
                if not self.workflow_config.workflow_fallback_enabled:
                    raise RuntimeError(f"Workflow execution failed with node failures: {failure_msg}")
                logger.warning(
                    "Workflow execution had node failures for session_id=%s: %s. Falling back to classic orchestration.",
                    session_id,
                    failure_msg,
                )
                self.tracking_adapter.log_params({"orchestration_fallback": "classic"})
                self.tracking_adapter.log_metrics({"workflow_fallback_count": 1})
                return self._answer_chat_with_orchestration(query, session_id)
            assistant_answer = response_state.generated_answer or "The workflow did not produce an answer."
            assistant_message = self.persistence_service.add_message(
                session_id=session_id,
                role="assistant",
                content=assistant_answer,
                metadata={
                    "retrieved_chunk_ids": [chunk.chunk_id for chunk in response_state.retrieved_chunks if chunk.chunk_id],
                },
            )
            return assistant_answer, user_message, assistant_message
        except Exception as exc:
            if self.workflow_config.workflow_fallback_enabled:
                logger.warning(
                    "Workflow execution failed for session_id=%s with error=%s. Falling back to classic orchestration.",
                    session_id,
                    str(exc),
                )
                self.tracking_adapter.log_params({"orchestration_fallback": "classic"})
                self.tracking_adapter.log_metrics({"workflow_fallback_count": 1})
                return self._answer_chat_with_orchestration(query, session_id)
            else:
                raise

    def answer_question_with_workflow(self, query: str) -> str:
        from workflow.state import WorkflowState

        workflow_id = str(uuid4())
        state = WorkflowState(query=query, workflow_execution_id=workflow_id, workflow_path="answer_question_with_workflow")
        response_state = self._run_workflow(state)
        return response_state.generated_answer or "The workflow did not produce an answer."

    def answer_chat_with_workflow(
        self,
        query: str,
        session_id: str,
    ) -> tuple[str, ChatMessageRecord, ChatMessageRecord]:
        if self.persistence_service is None:
            self.persistence_service = PersistenceService()
        user_message = self.persistence_service.add_message(
            session_id=session_id,
            role="user",
            content=query,
        )
        from workflow.state import WorkflowState

        workflow_id = str(uuid4())
        state = WorkflowState(
            query=query,
            session_id=session_id,
            message_id=user_message.message_id,
            workflow_execution_id=workflow_id,
            workflow_path="answer_chat_with_workflow",
        )
        response_state = self._run_workflow(state)
        assistant_answer = response_state.generated_answer or "The workflow did not produce an answer."
        assistant_message = self.persistence_service.add_message(
            session_id=session_id,
            role="assistant",
            content=assistant_answer,
            metadata={
                "retrieved_chunk_ids": [chunk.chunk_id for chunk in response_state.retrieved_chunks if chunk.chunk_id],
            },
        )
        return assistant_answer, user_message, assistant_message

    def _run_workflow(self, state: WorkflowState) -> WorkflowState:
        self.tracking_adapter.log_params(
            {
                "workflow_execution_id": state.workflow_execution_id,
                "workflow_path": state.workflow_path,
                "session_id": state.session_id or "none",
                "message_id": state.message_id or "none",
            }
        )
        try:
            response_state = self.workflow_graph.run(state)
        except Exception as exc:
            self.tracking_adapter.log_params(
                {
                    "workflow_failure": str(exc),
                    "workflow_path": state.workflow_path,
                }
            )
            self.tracking_adapter.log_metrics({"workflow_failure": 1})
            response_state = state
            response_state.generated_answer = (
                "The workflow could not complete the request. Please try again later."
            )
            response_state.citations = []
            response_state.orchestration_metrics = response_state.orchestration_metrics
        else:
            self._log_workflow_tracking(response_state)
        return response_state

    def _log_workflow_tracking(self, state: WorkflowState) -> None:
        self.tracking_adapter.log_params(
            {
                "workflow_execution_id": state.workflow_execution_id or "none",
                "workflow_path": state.workflow_path or "none",
                "prompt_name": state.prompt_template_used or "none",
                "prompt_version": state.prompt_version or "none",
                "retrieved_chunk_count": len(state.retrieved_chunks),
                "citation_count": len(state.citations),
                "workflow_node_failures": str(state.node_failures) if state.node_failures else "none",
            }
        )
        metrics: dict[str, float | int] = {
            "workflow_duration": state.workflow_duration or 0.0,
            "retrieved_chunk_count": len(state.retrieved_chunks),
            "citation_count": len(state.citations),
            "workflow_failure_count": len(state.node_failures),
        }
        for node_name, duration in state.node_timings.items():
            metrics[f"workflow_node_{node_name}_duration"] = duration
        self.tracking_adapter.log_metrics(metrics)

    def _generate_from_prompt(self, prompt: str) -> str:
        return self.generation_service.generate_from_prompt(prompt)
