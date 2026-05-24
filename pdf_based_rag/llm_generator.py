from collections.abc import Sequence

from models import RetrievedChunk
from services.generation_service import GenerationService, PROMPT_TEMPLATE


def generate_answer(query: str, context_chunks: Sequence[str | RetrievedChunk]) -> str:
    return GenerationService().generate_answer(query, context_chunks)
