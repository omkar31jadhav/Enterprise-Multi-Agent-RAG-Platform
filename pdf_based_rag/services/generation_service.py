from __future__ import annotations

from collections.abc import Sequence

from config import Config
from models import RetrievedChunk
from utils import get_logger


logger = get_logger(__name__)

PROMPT_TEMPLATE = """You are an intelligent AI tutor helping students learn from the provided study material.
Use only the context below to answer the question.
If the answer is not in the context, say that clearly.

Context:
{context}

Question:
{question}
"""


class GenerationService:
    def __init__(
        self,
        api_key: str | None = Config.GROQ_API_KEY,
        model: str = Config.LLM_MODEL,
        temperature: float = 0.2,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self._client = None

    def generate_answer(self, query: str, context_chunks: Sequence[str | RetrievedChunk]) -> str:
        if not context_chunks:
            return "No indexed document content was found. Ingest documents before asking questions."

        context = self._format_context(context_chunks)
        prompt = PROMPT_TEMPLATE.format(context=context, question=query)
        logger.info("Generating answer with model=%s context_chunks=%s", self.model, len(context_chunks))

        response = self._get_client().chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return str(response.choices[0].message.content.strip())

    def generate_from_prompt(self, prompt: str) -> str:
        if not prompt.strip():
            return "No indexed document content was found. Ingest documents before asking questions."

        logger.info("Generating answer from assembled prompt with model=%s", self.model)
        response = self._get_client().chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return str(response.choices[0].message.content.strip())

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise ValueError("Missing GROQ_API_KEY. Add it to your .env file before asking questions.")

        from groq import Groq

        self._client = Groq(api_key=self.api_key)
        return self._client

    @staticmethod
    def _format_context(context_chunks: Sequence[str | RetrievedChunk]) -> str:
        formatted_chunks: list[str] = []
        for chunk in context_chunks:
            if isinstance(chunk, RetrievedChunk):
                formatted_chunks.append(chunk.as_context())
            else:
                formatted_chunks.append(chunk)
        return "\n\n".join(formatted_chunks)
