from services.rag_service import RagService


def answer_user_question(query: str) -> str:
    return RagService().answer_question(query)
