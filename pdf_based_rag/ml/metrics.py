from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class OrchestrationTrackingMetrics:
    retrieval_time: float | None = None
    generation_time: float | None = None
    orchestration_total_time: float | None = None
    history_message_count: int = 0
    retrieved_chunk_count: int = 0
    citation_count: int = 0
    prompt_name: str = ""
    prompt_version: str = ""
    prompt_template_used: str = ""
    prompt_path: str = ""
    retrieval_scores: tuple[float | None, ...] = field(default_factory=tuple)
    context_size_estimate: int = 0


def serialize_metrics(metrics: OrchestrationTrackingMetrics) -> dict[str, Any]:
    payload = asdict(metrics)
    payload["retrieval_scores"] = json.dumps(
        [score if score is not None else None for score in metrics.retrieval_scores]
    )
    return payload
