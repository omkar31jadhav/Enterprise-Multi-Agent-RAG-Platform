from __future__ import annotations

import importlib
import sys
import types

from ml.evaluation import (
    evaluate_answer_grounding,
    evaluate_citation_coverage,
    evaluate_retrieval_relevance,
    evaluate_response_completeness,
)
from ml.metrics import OrchestrationTrackingMetrics, serialize_metrics
from ml.tracking import MLflowTrackingAdapter, NoOpTrackingAdapter, TrackingConfig
from models import RetrievedChunk
from rag.citations import CitationReference


def test_noop_tracking_adapter_is_safe() -> None:
    adapter = NoOpTrackingAdapter()
    adapter.start_run()
    adapter.log_params({"param": "value"})
    adapter.log_metrics({"metric": 1.0})
    adapter.set_tags({"tag": "value"})
    adapter.end_run()


def test_mlflow_tracking_adapter_invokes_mlflow_hooks(monkeypatch) -> None:
    fake_mlflow = types.SimpleNamespace(
        set_tracking_uri=lambda uri: setattr(fake_mlflow, "tracking_uri", uri),
        set_experiment=lambda name: setattr(fake_mlflow, "experiment", name),
        start_run=lambda run_name=None: setattr(fake_mlflow, "active", True),
        log_params=lambda params: setattr(fake_mlflow, "params", params),
        log_metrics=lambda metrics: setattr(fake_mlflow, "metrics", metrics),
        set_tags=lambda tags: setattr(fake_mlflow, "tags", tags),
        end_run=lambda: setattr(fake_mlflow, "active", False),
    )
    sys.modules["mlflow"] = fake_mlflow
    importlib.invalidate_caches()

    adapter = MLflowTrackingAdapter(
        TrackingConfig(
            experiment_name="test-experiment",
            run_name="run-1",
            tracking_uri="http://localhost:5000",
            enabled=True,
        )
    )
    adapter.start_run()
    adapter.log_params({"param": "value"})
    adapter.log_metrics({"metric": 2})
    adapter.set_tags({"tag": "value"})
    adapter.end_run()

    assert fake_mlflow.experiment == "test-experiment"
    assert fake_mlflow.tracking_uri == "http://localhost:5000"
    assert fake_mlflow.params == {"param": "value"}
    assert fake_mlflow.metrics == {"metric": 2.0}
    assert fake_mlflow.tags == {"tag": "value"}


def test_serialize_metrics_encodes_retrieval_scores_and_metadata() -> None:
    metrics = OrchestrationTrackingMetrics(
        retrieval_time=0.1,
        generation_time=0.2,
        orchestration_total_time=0.35,
        history_message_count=1,
        retrieved_chunk_count=2,
        citation_count=1,
        prompt_name="citation_aware_qa",
        prompt_version="v1",
        prompt_template_used="citation_aware_qa",
        prompt_path="rag_orchestrator",
        retrieval_scores=(0.8, None),
        context_size_estimate=150,
    )

    payload = serialize_metrics(metrics)

    assert payload["prompt_version"] == "v1"
    assert payload["retrieval_scores"] == "[0.8, null]"
    assert payload["context_size_estimate"] == 150


def test_evaluate_retrieval_and_response_scaffolding() -> None:
    chunks = [
        RetrievedChunk(content="text", source="doc.pdf", score=0.9),
        RetrievedChunk(content="other", source="doc.pdf", score=0.4),
    ]
    relevance = evaluate_retrieval_relevance(chunks, min_score=0.5)
    assert relevance.score == 0.5

    citations = [CitationReference(source="doc.pdf"), CitationReference(source="other.pdf")]
    coverage = evaluate_citation_coverage(citations, expected_sources=["doc.pdf", "missing.pdf"])
    assert coverage.coverage == 0.5

    grounding = evaluate_answer_grounding("Refer to doc.pdf for details.", chunks)
    assert grounding.grounding_score == 1.0

    completeness = evaluate_response_completeness("This is a complete answer.")
    assert completeness.completeness_score > 0.0
