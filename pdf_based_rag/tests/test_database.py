import pytest

from database.connection import Database
from database.models import DocumentRecord
from database.repositories import DocumentRepository, new_id
from database.services import PersistenceService
from models import DocumentChunk, RetrievedChunk, utc_now


def make_service(tmp_path) -> PersistenceService:
    database = Database(sqlite_path=tmp_path / "rag.db")
    return PersistenceService(database=database)


def test_database_initializes_schema(tmp_path) -> None:
    service = make_service(tmp_path)

    stats = service.get_stats()

    assert stats.document_count == 0
    assert stats.chunk_count == 0
    assert stats.chat_session_count == 0


def test_persistence_service_records_document_and_chunks(tmp_path) -> None:
    service = make_service(tmp_path)
    chunks = [
        DocumentChunk(
            content="battery safety instructions",
            document_name="manual.pdf",
            page_number=4,
            section_title="Battery Safety",
            chunk_index=0,
            metadata={"file_type": ".pdf"},
        )
    ]

    document = service.record_ingestion(
        filename="manual.pdf",
        file_type=".pdf",
        chunks=chunks,
        total_pages=12,
    )

    stats = service.get_stats()
    documents = service.list_documents()

    assert stats.document_count == 1
    assert stats.chunk_count == 1
    assert documents[0].id == document.id
    assert documents[0].filename == "manual.pdf"
    assert documents[0].total_pages == 12


def test_chat_persistence_records_session_and_messages(tmp_path) -> None:
    service = make_service(tmp_path)
    session = service.create_chat_session(user_identifier="tester")

    user_message = service.add_message(session.session_id, role="user", content="What is battery safety?")
    assistant_message = service.add_message(session.session_id, role="assistant", content="Use PPE.")
    messages = service.list_chat_messages(session.session_id)

    assert [message.message_id for message in messages] == [
        user_message.message_id,
        assistant_message.message_id,
    ]
    assert [message.role for message in messages] == ["user", "assistant"]


def test_retrieval_logging_preserves_scores_and_source_metadata(tmp_path) -> None:
    service = make_service(tmp_path)
    session = service.create_chat_session()
    message = service.add_message(session.session_id, role="user", content="battery?")

    logged_count = service.log_retrievals(
        query="battery?",
        session_id=session.session_id,
        message_id=message.message_id,
        chunks=[
            RetrievedChunk(
                content="battery safety instructions",
                source="manual.pdf",
                score=0.87,
                page_number=4,
                section_title="Battery Safety",
                chunk_id="chunk_123",
                metadata={"file_type": ".pdf"},
            )
        ],
    )
    retrievals = service.list_recent_retrievals(session_id=session.session_id)

    assert logged_count == 1
    assert retrievals[0].query == "battery?"
    assert retrievals[0].chunk_id == "chunk_123"
    assert retrievals[0].similarity_score == 0.87
    assert retrievals[0].document_name == "manual.pdf"
    assert retrievals[0].page_number == 4


def test_database_transaction_rolls_back_on_error(tmp_path) -> None:
    database = Database(sqlite_path=tmp_path / "rag.db")
    database.initialize()

    with pytest.raises(RuntimeError):
        with database.transaction() as connection:
            DocumentRepository(connection).create_document(
                DocumentRecord(
                    id=new_id("doc"),
                    filename="rollback.pdf",
                    file_type=".pdf",
                    ingestion_timestamp=utc_now(),
                )
            )
            raise RuntimeError("force rollback")

    service = PersistenceService(database=database, initialize=False)

    assert service.get_stats().document_count == 0
