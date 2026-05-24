from __future__ import annotations

from rag.prompts import CITATION_AWARE_QA_PROMPT, CONVERSATIONAL_QA_PROMPT, PromptTemplate


def should_use_conversational_prompt(history: str) -> bool:
    return bool(history and history.strip())


def select_prompt_template(history: str) -> PromptTemplate:
    if should_use_conversational_prompt(history):
        return CONVERSATIONAL_QA_PROMPT
    return CITATION_AWARE_QA_PROMPT
