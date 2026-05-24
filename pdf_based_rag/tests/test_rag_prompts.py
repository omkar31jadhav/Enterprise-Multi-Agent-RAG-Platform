from rag.prompts import CITATION_AWARE_QA_PROMPT, CONVERSATIONAL_QA_PROMPT, RETRIEVAL_QA_PROMPT, format_prompt


def test_retrieval_qa_prompt_formats_context_and_question() -> None:
    prompt = RETRIEVAL_QA_PROMPT.format(context="Document summary", question="What is the main point?")

    assert "Document summary" in prompt
    assert "What is the main point?" in prompt


def test_conversational_qa_prompt_requires_history_and_context() -> None:
    prompt = format_prompt(
        CONVERSATIONAL_QA_PROMPT,
        history="User: Hello\nAssistant: Hi there.",
        context="Relevant document context.",
        question="What should I focus on next?",
    )

    assert "Conversation history:" in prompt
    assert "Relevant document context." in prompt
    assert "What should I focus on next?" in prompt


def test_citation_aware_prompt_contains_source_guidance() -> None:
    prompt = CITATION_AWARE_QA_PROMPT.format(context="A short excerpt.", question="Give me a citation-aware answer.")

    assert "citation-aware" in prompt.lower()
    assert "A short excerpt." in prompt
    assert "Give me a citation-aware answer." in prompt
