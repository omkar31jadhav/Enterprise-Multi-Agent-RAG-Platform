from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class WorkflowConfig:
    """Runtime configuration for workflow orchestration execution."""

    workflow_enabled: bool = False
    workflow_fallback_enabled: bool = True
    workflow_observability_enabled: bool = True
    workflow_evaluation_enabled: bool = True

    @staticmethod
    def from_env() -> WorkflowConfig:
        """Load workflow configuration from environment variables."""
        return WorkflowConfig(
            workflow_enabled=os.getenv("WORKFLOW_ENABLED", "false").lower() in ("true", "1", "yes"),
            workflow_fallback_enabled=os.getenv("WORKFLOW_FALLBACK_ENABLED", "true").lower() in ("true", "1", "yes"),
            workflow_observability_enabled=os.getenv("WORKFLOW_OBSERVABILITY_ENABLED", "true").lower() in ("true", "1", "yes"),
            workflow_evaluation_enabled=os.getenv("WORKFLOW_EVALUATION_ENABLED", "true").lower() in ("true", "1", "yes"),
        )

    @staticmethod
    def disabled() -> WorkflowConfig:
        """Return a fully disabled workflow configuration."""
        return WorkflowConfig(
            workflow_enabled=False,
            workflow_fallback_enabled=False,
            workflow_observability_enabled=False,
            workflow_evaluation_enabled=True,  # Keep evaluation enabled for observability
        )

    @staticmethod
    def enabled_with_fallback() -> WorkflowConfig:
        """Return a configuration with workflow enabled and fallback safety."""
        return WorkflowConfig(
            workflow_enabled=True,
            workflow_fallback_enabled=True,
            workflow_observability_enabled=True,
            workflow_evaluation_enabled=True,
        )


__all__ = ["WorkflowConfig"]
