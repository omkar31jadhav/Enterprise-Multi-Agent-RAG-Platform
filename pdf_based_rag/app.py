import os
import tempfile

import streamlit as st

from config import Config
from database.services import PersistenceService
from services.generation_service import GenerationService
from services.ingestion_service import IngestionService
from services.rag_service import RagService
from services.retrieval_service import RetrievalService
from utils import configure_logging
from vector_db import get_vector_store

configure_logging()

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = None

st.title("PDF-Based RAG System")
st.subheader("Upload study materials and ask questions")
st.caption(f"Embedding model: `{Config.EMBEDDING_MODEL}`")

vector_store = get_vector_store()
persistence_service = PersistenceService()
if st.session_state.chat_session_id is None:
    st.session_state.chat_session_id = persistence_service.create_chat_session().session_id

ingestion_service = IngestionService(vector_store, persistence_service=persistence_service)
retrieval_service = RetrievalService(vector_store, persistence_service=persistence_service)
generation_service = GenerationService()
rag_service = RagService(
    retrieval_service=retrieval_service,
    generation_service=generation_service,
    persistence_service=persistence_service,
)
vector_stats = vector_store.collection_stats()

uploaded_files = st.file_uploader(
    "Upload documents (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

if uploaded_files:
    with st.spinner("Processing documents..."):
        for uploaded_file in uploaded_files:
            if uploaded_file.name in st.session_state.processed_files:
                continue

            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                file_path = tmp_file.name

            try:
                indexed_count = ingestion_service.ingest_file(file_path, source=uploaded_file.name)
                st.session_state.processed_files.add(uploaded_file.name)
                st.success(f"Processed {uploaded_file.name} with {indexed_count} chunks.")
            except Exception as exc:
                st.error(f"Error processing {uploaded_file.name}: {exc}")
            finally:
                os.unlink(file_path)

if vector_stats.record_count > 0:
    st.divider()
    st.subheader("Ask a question about your documents")

    for message in persistence_service.list_chat_messages(st.session_state.chat_session_id):
        with st.chat_message(message.role):
            st.write(message.content)

    question = st.chat_input("Ask a question about your documents")
    if question:
        with st.spinner("Searching for answers..."):
            try:
                answer, _, _ = rag_service.answer_chat(
                    question,
                    session_id=st.session_state.chat_session_id,
                )
                with st.chat_message("user"):
                    st.write(question)
                with st.chat_message("assistant"):
                    st.write(answer)
            except Exception as exc:
                st.error(f"Error generating answer: {exc}")
else:
    st.info("Upload documents or add files to the local documents folder to get started.")

st.sidebar.header("Configuration")
db_stats = persistence_service.get_stats()
st.sidebar.metric("Indexed Chunks", vector_stats.record_count)
st.sidebar.metric("Tracked Documents", db_stats.document_count)
st.sidebar.metric("Retrieval Events", db_stats.retrieval_count)
st.sidebar.caption(f"Vector backend: `{vector_stats.backend}`")
st.sidebar.caption(f"Collection: `{vector_stats.collection_name}`")
st.sidebar.caption(f"Chat session: `{st.session_state.chat_session_id}`")
if st.sidebar.button("Reset Documents"):
    vector_store.clear()
    st.session_state.processed_files = set()
    st.rerun()

if st.sidebar.button("New Chat"):
    st.session_state.chat_session_id = persistence_service.create_chat_session().session_id
    st.rerun()

with st.sidebar.expander("Documents"):
    for document in persistence_service.list_documents(limit=10):
        pages = f", {document.total_pages} pages" if document.total_pages is not None else ""
        st.caption(f"{document.filename} ({document.file_type}{pages})")

with st.sidebar.expander("Recent Retrievals"):
    for retrieval in persistence_service.list_recent_retrievals(
        limit=10,
        session_id=st.session_state.chat_session_id,
    ):
        score = (
            f"{retrieval.similarity_score:.4f}"
            if retrieval.similarity_score is not None
            else "n/a"
        )
        page = f" p.{retrieval.page_number}" if retrieval.page_number is not None else ""
        st.caption(f"{retrieval.document_name or 'unknown'}{page} | score {score}")
