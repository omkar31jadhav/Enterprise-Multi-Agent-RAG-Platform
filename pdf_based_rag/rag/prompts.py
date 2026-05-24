from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    template: str
    input_variables: tuple[str, ...] = ("context", "question")

    def format(self, **kwargs: str) -> str:
        missing = [key for key in self.input_variables if key not in kwargs]
        if missing:
            raise ValueError(f"Missing template variables: {missing}")
        return self.template.format(**kwargs)


RETRIEVAL_QA_PROMPT = PromptTemplate(
    name="retrieval_qa",
    template="""You are an enterprise knowledge assistant.
Use the information from the document context to answer the question accurately.
Cite the source, page number, and section if available.
If the answer is not present in the context, say that clearly.

Context:
{context}

Question:
{question}""",
)

CONVERSATIONAL_QA_PROMPT = PromptTemplate(
    name="conversational_qa",
    template="""You are an enterprise virtual assistant.
Maintain the conversation tone and use only the provided context.
If the answer is unknown, state that clearly.

Conversation history:
{history}

Context:
{context}

Question:
{question}""",
    input_variables=("history", "context", "question"),
)

CITATION_AWARE_QA_PROMPT = PromptTemplate(
    name="citation_aware_qa",
    template="""You are a compliance-focused assistant that provides citation-aware answers.
Answer the question using the document context and include source references.
List each reference on a separate line after the answer.

Context:
{context}

Question:
{question}""",
)


def format_prompt(template: PromptTemplate, **kwargs: str) -> str:
    return template.format(**kwargs)
