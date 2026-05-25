# AI Handoff Context

## 1. Project Overview

- Platform: PDF-based retrieval-augmented generation (RAG) for local documents.
- Purpose: ingest documents, persist metadata, retrieve semantically relevant chunks, and generate citation-aware answers.
- Engineering goals: preserve existing APIs, provide optional workflow orchestration, enforce typed boundaries, and surface observability.

## 2. Current Architecture

- `RagService`: primary facade for query/chat requests.
  - selects classic vs workflow path based on `WorkflowConfig` and `use_workflow` override.
  - logs metrics through `TrackingAdapter`.
  - falls back to classic orchestration when workflow fails and fallback is enabled.

- `RagOrchestrator`: lightweight classic orchestration.
  - loads conversational memory.
  - selects prompt template.
  - builds prompt from retrieved chunks + history.
  - calls generation via injected `generation_function`.
  - attaches citation references.
  - exposes `OrchestrationMetrics`.

- `WorkflowGraph`: optional typed node graph.
  - steps: `retrieval`, `memory`, `prompt_selection`, `generation`, `citation`, `evaluation`.
  - holds `RetrievalService`, `GenerationService`, `SQLConversationMemory`, `WorkflowConfig`.
  - populates `WorkflowState` and failure/timing metadata.

- workflow nodes: stateless responsibilities.
  - `retrieval_node`: query vector store.
  - `memory_node`: load prompt-ready history.
  - `prompt_selection_node`: choose prompt and format context.
  - `generation_node`: call generation service on assembled prompt.
  - `citation_node`: extract and attach citations.
  - `evaluation_node`: compute retrieval/grounding/completeness metrics.

- `RetrievalService`: typed vector search entrypoint.
  - delegates to `BaseVectorStore` / `ChromaVectorStore`.
  - logs retrievals via `PersistenceService`.

- `PersistenceService`: SQL-backed metadata and trace storage.
  - ingestion records, documents, chunks, chat sessions/messages.
  - retrieval logs, feedback, stats.

- `GenerationService`: model client wrapper.
  - uses `groq` chat API and `GROQ_API_KEY`.
  - generates from prompts or query+context.

- MLflow/evaluation infrastructure:
  - `TrackingAdapter` abstraction with `NoOpTrackingAdapter` and `MLflowTrackingAdapter`.
  - tracking logs params/metrics for retrieval, orchestration, workflow, and fallback.
  - evaluation node returns typed structures for retrieval relevance, citation coverage, answer grounding, and response completeness.

## 3. Workflow Architecture

- execution flow:
  1. `RagService.answer_question` or `answer_chat` creates `WorkflowState`.
  2. `WorkflowGraph.run` executes node sequence.
  3. state is enriched with retrieved chunks, history, prompt, answer, citations, metrics.
  4. failures are captured on the state; workflow may abort early.

- `WorkflowState`: typed state bag.
  - fields: `query`, `session_id`, `message_id`, `workflow_execution_id`, `workflow_path`, `retrieved_chunks`, `conversation_history`, `prompt`, `generated_answer`, `citations`, `orchestration_metrics`, `evaluation_results`, `prompt_template_used`, `prompt_version`, `workflow_duration`, `node_timings`, `node_failures`.

- node responsibilities:
  - retrieval: collect top-k chunks and optional filters.
  - memory: render prompt-ready history.
  - prompt selection: choose prompt template based on conversation history.
  - generation: call LLM and store answer.
  - citation: gather citation references from chunks and append to answer.
  - evaluation: compute analysis metrics.

- runtime toggles:
  - `WORKFLOW_ENABLED`: enable workflow execution.
  - `WORKFLOW_FALLBACK_ENABLED`: allow fallback to classic orchestration.
  - `WORKFLOW_OBSERVABILITY_ENABLED`: enable workflow observability.
  - `WORKFLOW_EVALUATION_ENABLED`: enable evaluation node.

- fallback behavior:
  - node failures are stored in `WorkflowState.node_failures`.
  - if `workflow_fallback_enabled` is true, the service logs fallback and runs classic orchestration.
  - if fallback is false, exceptions propagate.

## 4. Key Engineering Principles

- services are source-of-truth systems.
- workflow wraps existing services rather than replacing them.
- backward compatibility is prioritized.
- core orchestration architecture must remain loosely coupled.
- observability-first: metrics and logs are first-class outputs.
- additive changes preferred over invasive refactors.

## 5. Current Feature Status

- conversational memory: supported through SQL-backed `SQLConversationMemory`.
- citation-aware generation: supported in both classic and workflow paths.
- MLflow tracking: adapter pattern present; default is no-op.
- evaluation scaffolding: workflow evaluation node active.
- workflow orchestration: optional graph-based execution available.
- runtime toggles: workflow + fallback + evaluation configurable.
- fallback execution: workflow path can fall back to classic orchestration.
- typed metrics: `OrchestrationMetrics` and evaluation dataclasses present.
- mypy support: codebase type-checked.
- observability: logs, metrics, and node timings recorded.

## 6. Current Validation Status

- pytest: `60 passed`.
- mypy: passes cleanly under configured environment.
- compile validation: Python compileall passes excluding `.venv`.

## 7. Project Structure

- `services/`
  - `rag_service.py`
  - `retrieval_service.py`
  - `generation_service.py`
- `rag/`
  - `orchestration.py`
  - `memory.py`
  - `prompts.py`
  - `citations.py`
- `workflow/`
  - `graph.py`
  - `nodes.py`
  - `state.py`
  - `routing.py`
- `vectorstores/`
  - `chroma_store.py`
  - `base.py`
- `database/`
  - `services/persistence_service.py`
  - `models.py`
  - `repositories.py`
- `document_processing/`
- `embeddings/`
- `ml/`
  - `tracking.py`
  - `evaluation.py`
- `tests/`

## 8. Runtime Configuration

- `WORKFLOW_ENABLED`: enable workflow graph execution.
- `WORKFLOW_FALLBACK_ENABLED`: fall back to classic orchestration if workflow fails.
- `WORKFLOW_OBSERVABILITY_ENABLED`: enable workflow observability.
- `WORKFLOW_EVALUATION_ENABLED`: enable workflow evaluation node.
- `GROQ_API_KEY`: required for generation.
- `MLFLOW_*`: controls MLflow tracking when `MLflowTrackingAdapter` is used.

## 9. Workflow Execution Paths

- classic orchestration path:
  - `RagService` retrieves chunks via `RetrievalService`.
  - passes chunks to `RagOrchestrator.run`.
  - orchestrator loads history, builds prompt, generates answer, attaches citations.

- workflow orchestration path:
  - `RagService` builds `WorkflowState`.
  - `WorkflowGraph.run` executes nodes in order.
  - result is returned from `WorkflowState.generated_answer`.

- fallback flow:
  - if workflow node failures occur or exceptions are raised,
  - `RagService` logs fallback metrics,
  - and `RagService` delegates to classic orchestration when allowed.

## 10. Current Limitations / Intentional Non-Goals

- no autonomous agent loops / planning engines.
- no CrewAI or external agent orchestration.
- no async workflow execution.
- no distributed orchestration.
- no advanced branching or conditional workflow graphs.
- no multi-model routing or ensemble models.

## 11. Recommended Next Steps

1. add CI/CD workflow for `pytest`, `mypy`, and compile checks.
2. add GitHub Actions with `python -m pytest`, `python -m mypy`, and `python -m compileall`.
3. document runtime config and `.env` requirements.
4. verify production deployment path for `groq` model and SQLite persistence.
5. avoid multi-agent system work until core workflow stability is confirmed.

## 12. Important Constraints For Future AI Assistants

- do not rewrite the orchestration architecture.
- preserve service boundaries and public APIs.
- preserve backward compatibility for classic query/chat behavior.
- avoid coupling workflow nodes to external frameworks.
- prefer thin, additive workflow node changes.
- preserve typed state propagation and logging semantics.
