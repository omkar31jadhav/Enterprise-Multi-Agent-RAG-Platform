from config import Config
from config.workflow import WorkflowConfig
from database.services import PersistenceService
from ingest import ingest_documents
from services.rag_service import RagService
from utils import configure_logging

configure_logging()

print("\nPDF-based RAG system is initializing and indexing documents...")
indexed_chunks = ingest_documents(reset=True)

print(
    f"\nSystem is ready. Indexed {indexed_chunks} chunks from {Config.DOCUMENTS_DIR} and can answer questions now."
)

persistence_service = PersistenceService()
chat_session = persistence_service.create_chat_session(user_identifier="cli")
rag_service = RagService(
    persistence_service=persistence_service,
    workflow_config=WorkflowConfig.from_env(),
)
print(f"\nChat session: {chat_session.session_id}")

while True:
    query = input("\nYour Question (or type 'exit' to quit): ")
    if query.lower() in ["exit", "quit"]:
        print("\nExiting. Thanks for using the PDF-based RAG system.")
        break
    response, _, _ = rag_service.answer_chat(query, session_id=chat_session.session_id)
    print("\nAnswer:\n", response)
