"""
Workflow orchestration using LangGraph.

This module provides workflow-based orchestration that wraps existing services
while preserving backward compatibility with RagOrchestrator and RagService.
"""

from workflow.graph import build_workflow_graph
from workflow.state import WorkflowState

__all__ = [
    "WorkflowState",
    "build_workflow_graph",
]
